import numpy as np
from scipy.signal import butter, filtfilt, medfilt, resample
from config import config

def bandpass_filter(signal, fs, lowcut=None, highcut=None):
    """신호에 대역 통과 필터를 적용합니다."""
    if lowcut is None:
        lowcut = config.args.lowcut
    if highcut is None:
        highcut = config.args.highcut
        
    nyq = 0.5 * fs
    b, a = butter(N=2, Wn=[lowcut/nyq, highcut/nyq], btype='band')
    return filtfilt(b, a, signal)

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

def preprocess_ecg(ecg_signal, fs, target_fs=10):
    """ECG 신호를 전처리합니다."""
    # 대역 통과 필터 적용
    filtered = bandpass_filter(ecg_signal, fs)
    
    # 목표 샘플링 레이트로 다운샘플링
    num_samples = int(len(filtered) * (target_fs / fs))
    
    # 리샘플링
    return resample(filtered, num_samples)

def process_ecg_to_patches(ecg_signal, fs, patch_size, step):
    """ECG 신호를 처리하고 패치를 생성합니다."""
    # 전처리 적용
    processed = preprocess_ecg(ecg_signal, fs)
    
    # 패치 생성
    return patchify(processed, patch_size, step)

def preprocess_resp(resp_signal, fs, target_fs=5):
    """호흡 신호를 전처리합니다."""
    # 목표 샘플링 레이트로 다운샘플링
    num_samples = int(len(resp_signal) * (target_fs / fs))
    
    # 리샘플링
    resampled = resample(resp_signal, num_samples)
    
    # 메디안 필터 적용
    return medfilt(resampled, kernel_size=5)

def process_resp_to_patches(resp_signal, fs, patch_size, step):
    """호흡 신호를 처리하고 패치를 생성합니다."""
    # 전처리 적용
    processed = preprocess_resp(resp_signal, fs)
    
    # 패치 생성
    return patchify(processed, patch_size, step)

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