"""Robust respiratory-proxy extraction (v2).

This is an improved alternative to :func:`respiratory_extraction.resp_extraction_r`.
It keeps the phase-based motion magnification (PBM) front-end for ROI selection and
replaces the downstream signal extraction with soft-ROI Lucas-Kanade optical flow:

Improvements over v1 (resp_extraction_r)
-----------------------------------------
1. **Soft-ROI optical flow.** v1 tracks a single pixel; v2 picks the top-K
   highest-energy pixels from the PBM energy map and runs LK optical flow at
   each, then averages their y-displacements weighted by motion energy.  This
   makes the proxy far more robust to a single bad tracking point.
2. **Correct filter fps.** v1 halves the frame rate and then band-passes with
   a hard-coded ``fs=5`` that does not match the (≈2.5 Hz) sampling rate.
   v2 keeps every frame and band-passes with the actual ``fps``.
3. **Zero-phase filter.** v2 uses ``sosfiltfilt`` (zero-phase) instead of the
   causal ``sosfilt`` in v1, so the proxy has no phase delay relative to EDF.
4. **Per-epoch quality control.** A spectral SNR metric flags unreliable epochs.
"""

from pathlib import Path
import cv2
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


def _soft_roi_optical_flow(video, energy, top_k=10, bbx_half=25):
    """Soft-ROI LK optical flow: track top-K energy pixels, return weighted y-signal.

    Args:
        video: ``(T, H, W, 3)`` float array in ``[0, 1]``.
        energy: ``(H, W)`` PBM motion-energy map.
        top_k: number of seed points to track.
        bbx_half: half-size of the bounding box cropped around each seed point.

    Returns:
        1-D array of length T with energy-weighted mean y-displacement.
    """
    H, W = energy.shape
    flat_idx = np.argpartition(energy.ravel(), -top_k)[-top_k:]
    rows = flat_idx // W
    cols = flat_idx % W
    e_weights = energy.ravel()[flat_idx]
    e_weights = e_weights / e_weights.sum()

    uint8_frames = (np.clip(video, 0, 1) * 255).astype(np.uint8)
    T = len(uint8_frames)
    y_matrix = np.zeros((top_k, T), dtype=np.float32)

    for ki, (r, c, ew) in enumerate(zip(rows, cols, e_weights)):
        r0, r1 = max(0, r - bbx_half), min(H, r + bbx_half)
        c0, c1 = max(0, c - bbx_half), min(W, c + bbx_half)
        crop = uint8_frames[:, r0:r1, c0:c1]  # (T, h, w, 3)

        prev_gray = cv2.cvtColor(crop[0], cv2.COLOR_RGB2GRAY)
        cy = float(r - r0)
        pt = np.array([[[float(c - c0), cy]]], dtype=np.float32)

        y_track = [cy]
        for t in range(1, T):
            gray = cv2.cvtColor(crop[t], cv2.COLOR_RGB2GRAY)
            nxt, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, pt, None)
            if status is not None and status[0, 0] == 1:
                cy = float(nxt[0, 0, 1])
                pt = nxt
            y_track.append(cy)
            prev_gray = gray

        y_matrix[ki] = y_track

    # Energy-weighted mean y-position over time → (T,)
    raw = (y_matrix * e_weights[:, None]).sum(axis=0)
    return raw


def extract_respiratory_signal(
    video, fps,
    mag_factor, freq_range, attenuate, sigma, temporal_filter,
    roi_quantile=0.95, resp_band=(0.1, 0.5), top_k=10,
):
    """Extract a robust respiratory proxy from one epoch of video.

    Uses PBM to build a motion-energy ROI, then runs Lucas-Kanade optical flow
    at the top-K highest-energy seed points and averages their y-displacements
    weighted by motion energy.

    Args:
        video: ``(T, H, W, 3)`` float array in ``[0, 1]``.
        fps: true sampling rate of ``video`` (frames per second).
        mag_factor, freq_range, attenuate, sigma, temporal_filter: PBM params.
        roi_quantile: pixels at/above this energy quantile form the candidate ROI.
        resp_band: band-pass range in Hz.
        top_k: number of seed points to run optical flow on.

    Returns:
        ``(signal, max_point, energy_map, quality)``.
    """
    # --- Phase-based motion magnification (ROI selection only) ---
    no_mag, mag = motionMag(video, mag_factor, freq_range, attenuate, sigma, temporal_filter)

    # --- Spatial motion-energy map (H, W) ---
    energy = frame_difference(no_mag, mag)
    flat = np.argmax(energy)
    max_point = (int(flat // energy.shape[1]), int(flat % energy.shape[1]))

    # --- Soft-ROI optical flow: top-K energy seed points ---
    raw = _soft_roi_optical_flow(video, energy, top_k=top_k)

    # --- Zero-phase band-pass at the TRUE fps, then z-score ---
    filtered = _bandpass_zerophase(raw, fps, resp_band[0], resp_band[1])
    signal = _zscore(filtered)

    quality = respiratory_quality(signal, fps, resp_band)
    return signal, max_point, energy, quality


def resp_extraction_v2(
    video, fps, mag_factor, freq_range, attenuate, sigma, temporal_filter,
    save_dir, epoch_idx, preprocessed_signal=None, record_id=None,
    roi_quantile=0.95, resp_band=(0.1, 0.5), top_k=10,
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
        roi_quantile=roi_quantile, resp_band=resp_band, top_k=top_k,
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
