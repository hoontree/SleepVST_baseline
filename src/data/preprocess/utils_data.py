import numpy as np
from scipy.signal import butter, filtfilt, medfilt, resample, sosfilt
from sklearn.preprocessing import scale
import config

def normalize(patch):
    """패치를 정규화합니다 (평균 0, 표준편차 1)."""
    return (patch - np.mean(patch)) / (np.std(patch) + 1e-6)

def patchify(signal, patch_size, step):
    """신호에서 패치를 추출합니다."""
    patches = []
    for start in range(0, len(signal) - patch_size + 1, step):
        patch = signal[start:start + patch_size]
        patch = normalize(patch)
        patches.append(patch)
    return np.stack(patches)  # shape: (N, patch_size)

def bandpass_filter(signal, fs, lowcut=None, highcut=None):
    """신호에 대역 통과 필터를 적용합니다."""
    if lowcut is None:
        lowcut = args.lowcut
    if highcut is None:
        highcut = args.highcut
        
    nyq = 0.5 * fs
    b, a = butter(N=2, Wn=[lowcut/nyq, highcut/nyq], btype='band')
    return filtfilt(b, a, signal)

def preprocess_hw(ecg_signal, fs):
    '''
    Preprocess the ECG signal by applying a bandpass filter and resampling.
    '''
    # 데이터셋에 관계없이 동일한 전처리 로직 사용
    filtered = bandpass_filter(ecg_signal, fs)
    
    # 목표 샘플링 레이트: 10Hz
    target_fs = 10
    
    # 새 샘플 수 계산
    num_samples = int(len(filtered) * (target_fs / fs))
    
    # scipy.signal.resample 사용하여 다운샘플링
    return resample(filtered, num_samples)

# THOR RES → 5Hz Resampling - scipy.signal.resample 사용
def preprocess_bw(resp_signal, fs):
    '''
    Preprocess the THOR RES/Thor signal by applying resampling and median filtering.
    '''
    # 목표 샘플링 레이트: 5Hz
    target_fs = 5

    # 새 샘플 수 계산
    num_samples = int(len(resp_signal) * (target_fs / fs))

    # scipy.signal.resample 사용하여 다운샘플링
    resampled = resample(resp_signal, num_samples)

    # 메디안 필터 적용
    return medfilt(resampled, kernel_size=5)


# Respiratory signal processing functions (from respiratory_extraction.py)
def butter_bandpass_sos(lowcut, highcut, fs, order=1):
    '''Create butterworth bandpass filter using SOS format'''
    low = lowcut
    high = highcut
    sos = butter(order, [low, high], btype='band', fs=fs, output='sos', analog=False)
    return sos

def butter_bandpass_filter_sos(data, lowcut, highcut, fs, order=1):
    '''Apply butterworth bandpass filter using SOS format'''
    sos = butter_bandpass_sos(lowcut, highcut, fs, order=order)
    y = sosfilt(sos, data)
    return y

def filtered_signal_respiratory(signal, sample_freq, low=0.1, high=0.5, order=1):
    '''
    Apply butterworth bandpass filter for respiratory signals.
    Default frequency range: 0.1-0.5 Hz (6-30 breaths/min)
    '''
    filtered = butter_bandpass_filter_sos(signal, low, high, fs=sample_freq, order=order)
    return filtered

def normalize_respiratory(y_val):
    '''Normalize signal to [0, 1] range'''
    y = (y_val - np.min(y_val)) / (np.max(y_val) - np.min(y_val))
    return y

def processing_respiratory(signal, sample_freq=5, low=0.1, high=0.5, order=1):
    '''
    Complete respiratory signal processing pipeline:
    1. Standardization (mean=0, std=1)
    2. Butterworth bandpass filter
    3. Normalization to [0, 1]

    This mimics the processing pipeline from respiratory_extraction.py
    '''
    # Standardize signal
    standardized = scale(signal)

    # Apply butterworth bandpass filter
    filtered = filtered_signal_respiratory(standardized, sample_freq, low=low, high=high, order=order)

    # Normalize to [0, 1] range
    normalized = normalize_respiratory(filtered)

    return normalized

def preprocess_bw_respiratory(resp_signal, fs, target_fs=5, low=0.1, high=0.5, order=1):
    '''
    Preprocess respiratory signal using respiratory_extraction.py pipeline:
    1. Resample to target frequency (default 5Hz)
    2. Standardization
    3. Butterworth bandpass filter (default 0.1-0.5 Hz)
    4. Normalization to [0, 1]

    Args:
        resp_signal: Raw respiratory signal
        fs: Original sampling frequency
        target_fs: Target sampling frequency (default 5Hz)
        low: Low cutoff frequency for bandpass filter (default 0.1 Hz)
        high: High cutoff frequency for bandpass filter (default 0.5 Hz)
        order: Filter order (default 1)

    Returns:
        Processed respiratory signal
    '''
    # 1. Resample to target frequency
    num_samples = int(len(resp_signal) * (target_fs / fs))
    resampled = resample(resp_signal, num_samples)

    # 2-4. Apply respiratory signal processing pipeline
    processed = processing_respiratory(resampled, sample_freq=target_fs, low=low, high=high, order=order)

    return processed