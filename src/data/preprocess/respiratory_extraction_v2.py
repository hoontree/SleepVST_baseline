"""Robust respiratory-proxy extraction (v2).

This is an improved alternative to :func:`respiratory_extraction.resp_extraction_r`.
It keeps the phase-based motion magnification (PBM) front-end but replaces the
fragile parts of the downstream signal extraction:

Weaknesses of the legacy path and how v2 addresses them
-------------------------------------------------------
1. **Single max-point dependency.** The legacy code tracks the single pixel of
   maximum magnified-vs-original difference, so one large non-respiratory motion
   (body movement, blanket, lighting flicker) corrupts the whole epoch. v2 builds
   a *soft ROI* from the top-quantile motion-energy pixels and extracts a
   motion-energy-weighted spatial average, which is far less sensitive to a few
   outlier pixels.
2. **Method inconsistency.** The legacy code locates the point with PBM but then
   re-extracts the signal with Lucas-Kanade optical flow, mixing two different
   motion models. v2 uses a single, consistent signal model: luminance change of
   the magnified video over the ROI.
3. **Frame dropping + wrong filter fs.** The legacy code halves the frame rate
   (``make_half``) and then band-passes with a hard-coded ``fs=5`` that does not
   match the true sampling rate. v2 keeps every frame and band-passes with the
   actual ``fps`` using a zero-phase (``filtfilt``) filter.
4. **Vertical-only displacement.** Legacy tracks only the y displacement.
   Luminance change at moving edges responds to motion in any direction.
5. **No quality control.** v2 computes a respiratory-band quality metric
   (dominant breathing rate and spectral SNR) per epoch so downstream code can
   discard or down-weight unreliable epochs instead of trusting every output.

The function returns the extracted signal plus a quality dict; the pipeline is
responsible for persisting them.
"""

from pathlib import Path
import numpy as np
from scipy.signal import butter, sosfiltfilt, welch

from .motion_mag.phase_based import motionMag, frame_difference


def _bandpass_zerophase(signal, fs, low, high, order=2):
    """Zero-phase Butterworth band-pass.

    Uses ``sosfiltfilt`` so the filtered signal has no phase delay (important
    when the proxy is later aligned against a reference breathing waveform).
    Falls back to the raw signal when it is too short for the filter.
    """
    nyq = fs / 2.0
    low_n = max(low / nyq, 1e-4)
    high_n = min(high / nyq, 0.999)
    if high_n <= low_n:
        return signal - np.mean(signal)
    sos = butter(order, [low_n, high_n], btype='band', output='sos')
    # filtfilt needs more samples than the filter's padlen.
    padlen = 3 * (sos.shape[0] * 2)
    if len(signal) <= padlen:
        return signal - np.mean(signal)
    return sosfiltfilt(sos, signal)


def _zscore(x):
    std = np.std(x)
    if std < 1e-8:
        return x - np.mean(x)
    return (x - np.mean(x)) / std


def respiratory_quality(signal, fs, resp_band=(0.1, 0.5), peak_halfwidth=0.03):
    """Estimate respiratory-band quality of a 1-D signal.

    Returns a dict with:
      * ``breathing_rate``: dominant frequency in the band, in breaths/min.
      * ``snr``: power within ``±peak_halfwidth`` Hz of the dominant peak divided
        by the total power in ``resp_band`` (1.0 = perfectly monochromatic).
      * ``peak_freq``: dominant frequency in Hz.
      * ``ok``: ``True`` when the breathing rate is plausible and SNR is high
        enough to trust the epoch.
    """
    n = len(signal)
    if n < 8:
        return {"breathing_rate": float("nan"), "snr": 0.0,
                "peak_freq": float("nan"), "ok": False}

    nperseg = min(n, 256) if n < 256 else n  # one long window -> best resolution
    freqs, psd = welch(signal, fs=fs, nperseg=min(n, max(64, nperseg)))

    band = (freqs >= resp_band[0]) & (freqs <= resp_band[1])
    if not np.any(band) or psd[band].sum() <= 0:
        return {"breathing_rate": float("nan"), "snr": 0.0,
                "peak_freq": float("nan"), "ok": False}

    band_freqs = freqs[band]
    band_psd = psd[band]
    peak_freq = band_freqs[np.argmax(band_psd)]

    peak_win = np.abs(freqs - peak_freq) <= peak_halfwidth
    snr = float(psd[peak_win].sum() / psd[band].sum())
    breathing_rate = float(peak_freq * 60.0)

    ok = bool(snr >= 0.5 and 6.0 <= breathing_rate <= 30.0)
    return {"breathing_rate": breathing_rate, "snr": snr,
            "peak_freq": float(peak_freq), "ok": ok}


def extract_respiratory_signal(
    video, fps,
    mag_factor, freq_range, attenuate, sigma, temporal_filter,
    roi_quantile=0.95, resp_band=(0.1, 0.5),
):
    """Extract a robust respiratory proxy from one epoch of video.

    Args:
        video: ``(T, H, W, 3)`` float array in ``[0, 1]``.
        fps: true sampling rate of ``video`` (frames per second).
        mag_factor, freq_range, attenuate, sigma, temporal_filter: PBM params,
            passed through to :func:`motionMag`.
        roi_quantile: pixels whose motion energy is at/above this quantile form
            the ROI (e.g. 0.95 = top 5% most-moving pixels).
        resp_band: band-pass range in Hz for the respiratory signal.

    Returns:
        ``(signal, max_point, energy_map, quality)`` where ``signal`` is the
        z-scored band-passed proxy of length ``T``, ``max_point`` is the
        argmax-energy pixel (kept for visualization/compat), ``energy_map`` is
        the ``(H, W)`` motion-energy map, and ``quality`` is the dict from
        :func:`respiratory_quality`.
    """
    # --- Phase-based motion magnification ---
    no_mag, mag = motionMag(video, mag_factor, freq_range, attenuate, sigma, temporal_filter)

    # --- Spatial motion-energy map (H, W) ---
    energy = frame_difference(no_mag, mag)
    # argmax kept only for backward-compatible visualization
    flat = np.argmax(energy)
    max_point = (int(flat // energy.shape[1]), int(flat % energy.shape[1]))

    # --- Soft ROI: top-quantile energy pixels, energy-weighted ---
    thresh = np.quantile(energy, roi_quantile)
    weights = np.where(energy >= thresh, energy, 0.0)
    total_w = weights.sum()
    if total_w <= 0:
        # Degenerate (static) epoch: fall back to uniform weighting.
        weights = np.ones_like(energy)
        total_w = weights.sum()
    weights = weights / total_w

    # --- Respiratory signal: ROI-weighted luminance of the magnified video ---
    # Luminance change at moving edges encodes motion regardless of direction.
    mag_clipped = np.clip(mag, 0.0, 1.0)
    luminance = mag_clipped.mean(axis=-1)              # (T, H, W)
    raw = (luminance * weights[None, :, :]).sum(axis=(1, 2))  # (T,)

    # --- Zero-phase band-pass at the TRUE fps, then z-score ---
    filtered = _bandpass_zerophase(raw, fps, resp_band[0], resp_band[1])
    signal = _zscore(filtered)

    quality = respiratory_quality(signal, fps, resp_band)
    return signal, max_point, energy, quality


def resp_extraction_v2(
    video, fps, mag_factor, freq_range, attenuate, sigma, temporal_filter,
    save_dir, epoch_idx, preprocessed_signal=None, record_id=None,
    roi_quantile=0.95, resp_band=(0.1, 0.5),
    save_signals=True,
):
    """Pipeline-facing wrapper: extract, save proxy + quality, and plot.

    Saves:
      * ``epoch_{idx}_movement.npy``  – the respiratory proxy (compat filename).
      * ``epoch_{idx}_quality.npz``   – breathing rate, SNR, ok flag, max_point.

    Returns the quality dict so the caller can aggregate/skip low-quality epochs.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    signal, max_point, energy, quality = extract_respiratory_signal(
        video, fps, mag_factor, freq_range, attenuate, sigma, temporal_filter,
        roi_quantile=roi_quantile, resp_band=resp_band,
    )

    # Save proxy under the legacy filename so downstream consumers keep working.
    np.save(save_dir / f'epoch_{epoch_idx}_movement.npy', signal)
    # Save quality sidecar for epoch-level QC / filtering.
    np.savez(
        save_dir / f'epoch_{epoch_idx}_quality.npz',
        breathing_rate=quality["breathing_rate"],
        snr=quality["snr"],
        peak_freq=quality["peak_freq"],
        ok=quality["ok"],
        max_point=np.array(max_point),
    )

    if save_signals:
        _plot_signal(signal, fps, save_dir, epoch_idx, record_id, quality,
                     preprocessed_signal)

    return quality


def _plot_signal(signal, fps, save_dir, epoch_idx, record_id, quality,
                 preprocessed_signal=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    time = np.arange(len(signal)) / fps
    plt.figure(figsize=(10, 5))
    plt.plot(time, signal, label='Video proxy (v2)', color='blue', linewidth=1)
    if preprocessed_signal is not None:
        ref_t = np.arange(len(preprocessed_signal)) / fps
        # z-score the reference too so the two are on a comparable scale
        ref = preprocessed_signal - np.mean(preprocessed_signal)
        ref_std = np.std(ref)
        if ref_std > 1e-8:
            ref = ref / ref_std
        plt.plot(ref_t, ref, label='Reference (EDF)', color='orange', linewidth=1)
        plt.legend(loc='upper right', fontsize=10)

    flag = "OK" if quality["ok"] else "LOW-Q"
    plt.title(f'Record {record_id} - Epoch {epoch_idx} - '
              f'BR={quality["breathing_rate"]:.1f}/min  SNR={quality["snr"]:.2f}  [{flag}]',
              fontsize=12)
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Signal (z)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.savefig(str(save_dir / f'signal_epoch_{epoch_idx}.png'), bbox_inches='tight', dpi=150)
    plt.close()
