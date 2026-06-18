"""Signal preprocessing utilities.

This module provides 1-D biosignal (ECG, respiratory, etc.) preprocessing
functions used throughout the SleepVST pipeline. Its main capabilities are:

* Patch-wise segmentation and z-normalization (:func:`patchify`,
  :func:`normalize`).
* Butterworth band-pass filtering (:func:`bandpass_filter`,
  :func:`butter_bandpass_filter_sos`).
* Resampling and filtering of ECG/respiratory signals
  (:func:`preprocess_hw`, :func:`preprocess_bw`).
* Respiratory signal processing that reproduces the
  ``respiratory_extraction.py`` pipeline
  (:func:`processing_respiratory`, :func:`preprocess_bw_respiratory`).
"""

import numpy as np
from scipy.signal import butter, filtfilt, medfilt, resample, sosfilt
from sklearn.preprocessing import scale

# Default band-pass cutoff frequencies (Hz) for ECG preprocessing.
# These match the historical CLI defaults (--lowcut / --highcut).
DEFAULT_LOWCUT = 0.66
DEFAULT_HIGHCUT = 2.8

def normalize(patch):
    """Z-normalize a single patch.

    Subtracts the mean and divides by the standard deviation so the patch has
    approximately zero mean and unit standard deviation. A small epsilon
    (``1e-6``) is added to the denominator to avoid division by zero when the
    standard deviation is close to zero.

    Args:
        patch (numpy.ndarray): 1-D signal patch to normalize.

    Returns:
        numpy.ndarray: Normalized patch with the same shape as the input.
    """
    return (patch - np.mean(patch)) / (np.std(patch) + 1e-6)

def patchify(signal, patch_size, step):
    """Split a 1-D signal into fixed-length, normalized patches.

    Slides a window across the signal with a stride of ``step``, extracting
    patches of length ``patch_size`` and z-normalizing each one with
    :func:`normalize`. Any trailing samples that cannot fill a full
    ``patch_size`` window are discarded.

    Args:
        signal (numpy.ndarray): 1-D input signal to segment.
        patch_size (int): Length of each patch in samples.
        step (int): Stride between consecutive patch start positions, in
            samples.

    Returns:
        numpy.ndarray: Array of patches with shape ``(N, patch_size)``, where
        ``N = (len(signal) - patch_size) // step + 1``.

    Raises:
        ValueError: Raised by :func:`numpy.stack` when no patch can be
            extracted (e.g. ``len(signal) < patch_size``).
    """
    patches = []
    for start in range(0, len(signal) - patch_size + 1, step):
        patch = signal[start:start + patch_size]
        patch = normalize(patch)
        patches.append(patch)
    return np.stack(patches)  # shape: (N, patch_size)

def bandpass_filter(signal, fs, lowcut=None, highcut=None):
    """Apply a 2nd-order Butterworth band-pass filter to a signal.

    Uses :func:`scipy.signal.filtfilt` for zero-phase filtering, so no phase
    delay is introduced. When ``lowcut`` or ``highcut`` is not provided, the
    module defaults (:data:`DEFAULT_LOWCUT`, :data:`DEFAULT_HIGHCUT`) are used.

    Args:
        signal (numpy.ndarray): 1-D input signal to filter.
        fs (float): Sampling frequency of the input signal, in Hz.
        lowcut (float, optional): Lower cutoff frequency in Hz. Defaults to
            :data:`DEFAULT_LOWCUT` when ``None``.
        highcut (float, optional): Upper cutoff frequency in Hz. Defaults to
            :data:`DEFAULT_HIGHCUT` when ``None``.

    Returns:
        numpy.ndarray: Band-pass filtered signal with the same length as the
        input.
    """
    if lowcut is None:
        lowcut = DEFAULT_LOWCUT
    if highcut is None:
        highcut = DEFAULT_HIGHCUT

    nyq = 0.5 * fs
    b, a = butter(N=2, Wn=[lowcut/nyq, highcut/nyq], btype='band')
    return filtfilt(b, a, signal)

def preprocess_hw(ecg_signal, fs):
    """Band-pass filter an ECG (heartbeat) signal and resample it to 10 Hz.

    Applies the same preprocessing logic regardless of the source dataset. The
    signal is first band-pass filtered with :func:`bandpass_filter` and then
    downsampled to the target sampling rate (10 Hz) using
    :func:`scipy.signal.resample`.

    Args:
        ecg_signal (numpy.ndarray): Raw ECG signal.
        fs (float): Sampling frequency of the input signal, in Hz.

    Returns:
        numpy.ndarray: Filtered signal resampled to 10 Hz.
    """
    # Use the same preprocessing logic regardless of the dataset.
    filtered = bandpass_filter(ecg_signal, fs)

    # Target sampling rate: 10 Hz
    target_fs = 10

    # Compute the new number of samples.
    num_samples = int(len(filtered) * (target_fs / fs))

    # Downsample using scipy.signal.resample.
    return resample(filtered, num_samples)

def preprocess_bw(resp_signal, fs):
    """Resample a respiratory (THOR RES) signal to 5 Hz and median-filter it.

    Downsamples the signal to the target sampling rate (5 Hz) using
    :func:`scipy.signal.resample`, then applies a median filter with a kernel
    size of 5 to remove impulsive noise.

    Args:
        resp_signal (numpy.ndarray): Raw respiratory signal (THOR RES/Thor).
        fs (float): Sampling frequency of the input signal, in Hz.

    Returns:
        numpy.ndarray: Signal resampled to 5 Hz and median-filtered.
    """
    # Target sampling rate: 5 Hz
    target_fs = 5

    # Compute the new number of samples.
    num_samples = int(len(resp_signal) * (target_fs / fs))

    # Downsample using scipy.signal.resample.
    resampled = resample(resp_signal, num_samples)

    # Apply a median filter.
    return medfilt(resampled, kernel_size=5)


# Respiratory signal processing functions (from respiratory_extraction.py)
def butter_bandpass_sos(lowcut, highcut, fs, order=1):
    """Create Butterworth band-pass filter coefficients in SOS format.

    Returns numerically stable second-order sections (SOS) filter
    coefficients. Cutoff frequencies are given as absolute frequencies (Hz)
    and are normalized internally by :func:`scipy.signal.butter` via the
    ``fs`` argument.

    Args:
        lowcut (float): Lower cutoff frequency in Hz.
        highcut (float): Upper cutoff frequency in Hz.
        fs (float): Sampling frequency of the signal, in Hz.
        order (int, optional): Filter order. Defaults to 1.

    Returns:
        numpy.ndarray: SOS filter coefficients with shape ``(n_sections, 6)``.
    """
    low = lowcut
    high = highcut
    sos = butter(order, [low, high], btype='band', fs=fs, output='sos', analog=False)
    return sos

def butter_bandpass_filter_sos(data, lowcut, highcut, fs, order=1):
    """Apply a Butterworth band-pass filter in SOS format to a signal.

    Builds the filter coefficients with :func:`butter_bandpass_sos` and
    performs one-directional (causal) filtering with
    :func:`scipy.signal.sosfilt`. This is not zero-phase filtering, so the
    output contains a phase delay.

    Args:
        data (numpy.ndarray): 1-D input signal to filter.
        lowcut (float): Lower cutoff frequency in Hz.
        highcut (float): Upper cutoff frequency in Hz.
        fs (float): Sampling frequency of the signal, in Hz.
        order (int, optional): Filter order. Defaults to 1.

    Returns:
        numpy.ndarray: Filtered signal with the same length as the input.
    """
    sos = butter_bandpass_sos(lowcut, highcut, fs, order=order)
    y = sosfilt(sos, data)
    return y

def filtered_signal_respiratory(signal, sample_freq, low=0.1, high=0.5, order=1):
    """Apply a Butterworth band-pass filter tuned for respiratory signals.

    The default pass band is 0.1-0.5 Hz, corresponding to a normal respiratory
    rate of 6-30 breaths per minute.

    Args:
        signal (numpy.ndarray): 1-D respiratory signal to filter.
        sample_freq (float): Sampling frequency of the signal, in Hz.
        low (float, optional): Lower cutoff frequency in Hz. Defaults to 0.1.
        high (float, optional): Upper cutoff frequency in Hz. Defaults to 0.5.
        order (int, optional): Filter order. Defaults to 1.

    Returns:
        numpy.ndarray: Band-pass filtered respiratory signal.
    """
    filtered = butter_bandpass_filter_sos(signal, low, high, fs=sample_freq, order=order)
    return filtered

def normalize_respiratory(y_val):
    """Min-max normalize a signal to the [0, 1] range.

    Args:
        y_val (numpy.ndarray): Input signal to normalize.

    Returns:
        numpy.ndarray: Signal scaled to the [0, 1] range with the same shape as
        the input.

    Note:
        When all values are identical (``max == min``), this divides by zero
        and the result may contain ``NaN`` or ``inf``.
    """
    y = (y_val - np.min(y_val)) / (np.max(y_val) - np.min(y_val))
    return y

def processing_respiratory(signal, sample_freq=5, low=0.1, high=0.5, order=1):
    """Run the full processing pipeline on a respiratory signal.

    Reproduces the processing flow of ``respiratory_extraction.py`` in three
    steps:

    1. Standardization (zero mean, unit standard deviation) via
       :func:`sklearn.preprocessing.scale`.
    2. Butterworth band-pass filtering via
       :func:`filtered_signal_respiratory`.
    3. Normalization to the [0, 1] range via :func:`normalize_respiratory`.

    Args:
        signal (numpy.ndarray): 1-D respiratory signal to process.
        sample_freq (float, optional): Sampling frequency of the signal, in Hz.
            Defaults to 5.
        low (float, optional): Lower cutoff frequency in Hz. Defaults to 0.1.
        high (float, optional): Upper cutoff frequency in Hz. Defaults to 0.5.
        order (int, optional): Filter order. Defaults to 1.

    Returns:
        numpy.ndarray: Respiratory signal after standardization, filtering, and
        normalization.
    """
    # Standardize signal
    standardized = scale(signal)

    # Apply butterworth bandpass filter
    filtered = filtered_signal_respiratory(standardized, sample_freq, low=low, high=high, order=order)

    # Normalize to [0, 1] range
    normalized = normalize_respiratory(filtered)

    return normalized

def preprocess_bw_respiratory(resp_signal, fs, target_fs=5, low=0.1, high=0.5, order=1):
    """Resample a raw respiratory signal and run the respiratory pipeline.

    Based on the ``respiratory_extraction.py`` pipeline, processing proceeds in
    the following order:

    1. Resample to the target frequency (default 5 Hz) via
       :func:`scipy.signal.resample`.
    2. Standardization.
    3. Butterworth band-pass filtering (default 0.1-0.5 Hz).
    4. Normalization to the [0, 1] range.

    Steps 2-4 are delegated to :func:`processing_respiratory`.

    Args:
        resp_signal (numpy.ndarray): Raw respiratory signal.
        fs (float): Original sampling frequency of the input signal, in Hz.
        target_fs (float, optional): Target sampling frequency in Hz. Defaults
            to 5.
        low (float, optional): Lower cutoff frequency of the band-pass filter,
            in Hz. Defaults to 0.1.
        high (float, optional): Upper cutoff frequency of the band-pass filter,
            in Hz. Defaults to 0.5.
        order (int, optional): Filter order. Defaults to 1.

    Returns:
        numpy.ndarray: Respiratory signal after resampling and the processing
        pipeline.
    """
    # 1. Resample to target frequency
    num_samples = int(len(resp_signal) * (target_fs / fs))
    resampled = resample(resp_signal, num_samples)

    # 2-4. Apply respiratory signal processing pipeline
    processed = processing_respiratory(resampled, sample_freq=target_fs, low=low, high=high, order=order)

    return processed
