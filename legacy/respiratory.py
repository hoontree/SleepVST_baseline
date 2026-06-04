import cv2
import numpy as np
from scipy.signal import welch, butter, filtfilt, detrend
from scipy.fft import fft2, ifft2
from typing import Tuple, List, Dict, Optional, Set
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
from common.logger import Logger
import os
import time
import gc  # 가비지 컬렉션 추가
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import concurrent.futures
import signal
import psutil  # 프로세스 관리 추가
import traceback  # 에러 traceback 추가

class RespiratorSignalProcessor:
    """호흡 신호 처리를 위한 메인 클래스"""
    
    def __init__(self, 
                 block_size: Tuple[int, int] = (16, 40),
                 overlap: int = 8,
                 delta_t: int = 3,
                 resp_band: Tuple[int, int] = (5, 60),
                 window_sec: int = 30,
                 cumulative_threshold: float = 0.4,  # 누적 가중치 임계값 (80-90%)
                 initial_windows: int = 5):  # 초기 150초 = 5개 윈도우
        self.block_size = block_size
        self.overlap = overlap
        self.delta_t = delta_t
        self.resp_band = resp_band
        self.window_sec = window_sec
        self.cumulative_threshold = cumulative_threshold
        self.initial_windows = initial_windows
        
        # 상태 관리
        self.window_count = 0
        self.selected_blocks_history = []
        self.logger = Logger(dir='logs', name='RespiratorySignalProcessor.log')
        
    def compute_snr(self, resp_signal: np.ndarray, fs: float, window_size: int) -> float:
        """
        SNR 계산
        
        Args:
            resp_signal: 호흡 신호 (1차원 numpy array)
            fs: 샘플링 주파수 (Hz)
            window_size: 윈도우 길이 (frame 수)
        
        Returns:
            float: SNR 값
        """
        if len(resp_signal) < window_size:
            return 0.0
            
        freqs, psd = welch(resp_signal, fs=fs, nperseg=min(window_size, len(resp_signal)))
        
        low_f, high_f = self.resp_band[0]/60, self.resp_band[1]/60
        mask = (freqs >= low_f) & (freqs <= high_f)
        
        if not np.any(mask):
            return 0.0
            
        signal_power = np.max(psd[mask])
        total_power = np.sum(psd[mask])
        
        return signal_power / total_power if total_power > 0 else 0.0

    def _setup_video_capture(self, video_path: str, max_frames: Optional[int] = None) -> Tuple[cv2.VideoCapture, float, int, Tuple[int, int]]:
        """비디오 캡처 설정"""
        cap = cv2.VideoCapture(video_path)
        
        ret, frame = cap.read()
        if not ret:
            raise ValueError("비디오를 읽을 수 없거나 비어있습니다.")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        h, w = frame.shape[:2]
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - self.delta_t
        if max_frames is None:
            max_frames = total_frames
        else:
            max_frames = min(max_frames, total_frames)
            
        # 첫 번째 프레임으로 되돌리기
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        return cap, fps, max_frames, (h, w)

    def _create_filter(self, fps: float) -> Tuple[np.ndarray, np.ndarray]:
        """대역통과 필터 생성"""
        nyq = fps / 2
        low, high = 0.1 / nyq, 0.7 / nyq
        return butter(2, [low, high], btype='band')

    def _generate_all_block_positions(self, frame_shape: Tuple[int, int]) -> List[Tuple[int, int]]:
        """모든 블록 위치 생성"""
        h, w = frame_shape
        block_h, block_w = self.block_size
        step_h = block_h - self.overlap
        
        return [
            (y, x)
            for y in range(0, h - block_h + 1, step_h)
            for x in range(0, w - block_w + 1, block_w)
        ]

    def _get_neighbor_positions(self, 
                              selected_blocks: List[Tuple[int, int]], 
                              all_positions: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        선택된 블록들 주변의 3x3 이웃 블록들을 후보로 선정
        
        Args:
            selected_blocks: 이전에 선택된 블록들
            all_positions: 전체 블록 위치 리스트
            
        Returns:
            List[Tuple[int, int]]: 후보 블록 위치들 (중복 제거됨)
        """
        if not selected_blocks:
            return all_positions
            
        candidate_set = set()
        block_h, block_w = self.block_size
        step_h = block_h - self.overlap
        
        for y, x in selected_blocks:
            # 3x3 이웃 영역 생성
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    neighbor_y = y + dy * step_h
                    neighbor_x = x + dx * block_w
                    neighbor_pos = (neighbor_y, neighbor_x)
                    
                    # 유효한 위치인지 확인
                    if neighbor_pos in all_positions:
                        candidate_set.add(neighbor_pos)
        
        return list(candidate_set)

    def _select_blocks_by_cumulative_threshold(self, 
                                             snrs: List[Tuple[Tuple[int, int], float]]) -> List[Tuple[int, int]]:
        """
        누적 가중치 기준으로 블록 선택
        
        Args:
            snrs: (position, snr_value) 튜플 리스트
            
        Returns:
            List[Tuple[int, int]]: 선택된 블록 위치들
        """
        if not snrs:
            return []
        
        # SNR 내림차순 정렬
        sorted_snrs = sorted(snrs, key=lambda x: x[1], reverse=True)
        
        # 양수 SNR만 고려
        positive_snrs = [(pos, snr) for pos, snr in sorted_snrs if snr > 0]
        if not positive_snrs:
            return []
        
        # 전체 SNR 합 계산
        total_snr = sum(snr for _, snr in positive_snrs)
        if total_snr <= 0:
            return []
        
        # 누적 SNR 합이 임계값에 도달할 때까지 블록 선택
        cumulative_snr = 0
        selected_blocks = []
        
        for pos, snr in positive_snrs:
            cumulative_snr += snr
            selected_blocks.append(pos)
            
            # 누적 비율이 임계값을 넘으면 종료
            if cumulative_snr / total_snr >= self.cumulative_threshold:
                break
        
        # 최소 1개는 선택하도록 보장
        if not selected_blocks and positive_snrs:
            selected_blocks = [positive_snrs[0][0]]
        
        return selected_blocks

    def _process_block_motion(self, b1: np.ndarray, b2: np.ndarray) -> float:
        """블록 간 모션 처리 (cross-correlation 기반)"""
        if b1.size == 0 or b2.size == 0:
            return 0.0
            
        # DCNorm
        b1_mean = b1.mean()
        b2_mean = b2.mean()
        
        if b1_mean == 0 or b2_mean == 0:
            return 0.0
            
        b1_norm = b1 / b1_mean - 1.0
        b2_norm = b2 / b2_mean - 1.0
        
        # FFT 기반 cross-correlation
        F1 = fft2(b1_norm)
        F2 = fft2(b2_norm)
        C = np.real(ifft2(ifft2(F1 * np.conj(F2))))
        
        # 피크 찾기
        ypeak, xpeak = np.unravel_index(np.argmax(C), C.shape)
        
        # Sub-pixel 정밀도를 위한 보간
        return self._calculate_subpixel_shift(C, ypeak, xpeak)

    def _calculate_subpixel_shift(self, correlation: np.ndarray, y0: int, x0: int) -> float:
        """Sub-pixel shift 계산"""
        col = correlation[:, x0]
        
        if 1 <= y0 < col.size - 1:
            f0 = col[y0]
            f1 = col[y0 + 1]
            f_1 = col[y0 - 1]
            denom = 2 * (2 * f0 - f1 - f_1)
            
            if denom != 0:
                return y0 + (f1 - f_1) / denom
                
        return float(y0)

    def _load_frame_buffers(self, cap: cv2.VideoCapture, frames_to_load: int) -> List[np.ndarray]:
        """프레임 버퍼 로드"""
        buffers = []
        for _ in range(frames_to_load):
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
            buffers.append(gray)
        return buffers

    def _process_traces_with_progress(self, 
                                    traces: Dict[Tuple[int, int], List[float]], 
                                    candidate_positions: List[Tuple[int, int]], 
                                    b: np.ndarray, 
                                    a: np.ndarray,
                                    window_frames: int,
                                    fps: float) -> Tuple[List[Tuple[Tuple[int, int], float]], Dict[Tuple[int, int], np.ndarray]]:
        """트레이스 처리 및 SNR 계산 (진행률 표시)"""
        snrs = []
        block_signal = {}
        
        for pos in candidate_positions:
            if pos not in traces or len(traces[pos]) == 0:
                continue
                
            # 누적합을 통한 변위 신호 생성
            displacement_signal = np.cumsum(traces[pos])
            
            if len(displacement_signal) == 0:
                continue
                
            # 신호 전처리
            detrended = detrend(displacement_signal)
            
            # 필터링 (길이가 충분한 경우에만)
            if len(detrended) > 6:  # 필터 차수보다 긴 경우
                try:
                    filtered = filtfilt(b, a, detrended)
                except:
                    filtered = detrended
            else:
                filtered = detrended
            
            # Z-스코어 정규화 및 클리핑
            if len(filtered) > 0 and np.std(filtered) > 0:
                z_score = (filtered - np.mean(filtered)) / np.std(filtered)
                clipped = np.clip(z_score, -3, 3)
            else:
                clipped = np.zeros_like(filtered) if len(filtered) > 0 else np.array([0.0])
            
            # 최근 윈도우 크기만큼 유지
            block_signal[pos] = clipped[-window_frames:] if len(clipped) >= window_frames else clipped
            
            # SNR 계산
            if len(block_signal[pos]) > 0:
                snr_val = self.compute_snr(clipped, fps, len(clipped))
                snrs.append((pos, snr_val))
        
        return snrs, block_signal

    def _process_traces(self, 
                       traces: Dict[Tuple[int, int], List[float]], 
                       candidate_positions: List[Tuple[int, int]], 
                       b: np.ndarray, 
                       a: np.ndarray,
                       window_frames: int,
                       fps: float) -> Tuple[List[Tuple[Tuple[int, int], float]], Dict[Tuple[int, int], np.ndarray]]:
        """트레이스 처리 및 SNR 계산 (진행률 표시 없음)"""
        snrs = []
        block_signal = {}
        
        for pos in candidate_positions:
            if pos not in traces or len(traces[pos]) == 0:
                continue
                
            # 누적합을 통한 변위 신호 생성
            displacement_signal = np.cumsum(traces[pos])
            
            if len(displacement_signal) == 0:
                continue
                
            # 신호 전처리
            detrended = detrend(displacement_signal)
            
            # 필터링 (길이가 충분한 경우에만)
            if len(detrended) > 6:  # 필터 차수보다 긴 경우
                try:
                    filtered = filtfilt(b, a, detrended)
                except:
                    filtered = detrended
            else:
                filtered = detrended
            
            # Z-스코어 정규화 및 클리핑
            if len(filtered) > 0 and np.std(filtered) > 0:
                z_score = (filtered - np.mean(filtered)) / np.std(filtered)
                clipped = np.clip(z_score, -3, 3)
            else:
                clipped = np.zeros_like(filtered) if len(filtered) > 0 else np.array([0.0])
            
            # 최근 윈도우 크기만큼 유지
            block_signal[pos] = clipped[-window_frames:] if len(clipped) >= window_frames else clipped
            
            # SNR 계산
            if len(block_signal[pos]) > 0:
                snr_val = self.compute_snr(clipped, fps, len(clipped))
                snrs.append((pos, snr_val))
        
        return snrs, block_signal

    def _combine_signals(self, 
                        selected_blocks: List[Tuple[int, int]], 
                        block_signal: Dict[Tuple[int, int], np.ndarray], 
                        snrs: List[Tuple[Tuple[int, int], float]]) -> np.ndarray:
        """선택된 블록들의 신호를 가중평균으로 결합"""
        if not selected_blocks:
            return np.array([0.0])
        
        # SNR 가중치 생성
        snr_dict = dict(snrs)
        weights = np.array([snr_dict.get(pos, 0) for pos in selected_blocks])
        
        if np.sum(weights) == 0:
            weights = np.ones(len(weights))
        else:
            weights = weights / np.sum(weights)
        
        # 신호 수집
        signals = [block_signal[pos] for pos in selected_blocks if pos in block_signal and len(block_signal[pos]) > 0]
        
        if not signals:
            return np.array([0.0])
        
        # 모든 신호를 같은 길이로 맞춤
        min_length = min(len(sig) for sig in signals)
        if min_length == 0:
            return np.array([0.0])
            
        aligned_signals = [sig[-min_length:] for sig in signals]
        
        # 가중평균 계산
        weights = weights[:len(aligned_signals)]
        weights += 1e-8  # 수치적 안정성
        weights = weights / np.sum(weights)  # 재정규화
        
        return np.average(aligned_signals, axis=0, weights=weights)

    def extract_respiratory_signal(self, 
                                 video_path: Path,
                                 max_frames: Optional[int] = None) -> np.ndarray:
        """
        Auto-RoI를 사용한 호흡 신호 추출
        
        Args:
            video_path: 비디오 파일 경로
            max_frames: 처리할 최대 프레임 수
            
        Returns:
            np.ndarray: 추출된 호흡 신호
        """
        cap = None
        try:
            # 비디오 설정
            cap, fps, max_frames, frame_shape = self._setup_video_capture(str(video_path), max_frames)
            
            # 필터 및 위치 설정
            b, a = self._create_filter(fps)
            all_positions = self._generate_all_block_positions(frame_shape)
            window_frames = int(self.window_sec * fps)
            
            # 초기화
            traces = {pos: [] for pos in all_positions}
            selected_blocks = []
            combined_signal = []
            
            frame_idx = 0
            frames_since_update = 0
            self.window_count = 0
            
            total_windows = max_frames // window_frames + 1

            self.logger.info(f"{video_path.stem} 전체 블록 수: {len(all_positions)}, 윈도우 크기: {window_frames} 프레임")
            self.logger.info(f"{video_path.stem} 예상 총 윈도우 수: {total_windows}")

            while frame_idx < max_frames:
                # 프레임 로드
                frames_to_load = min(window_frames, max_frames - frame_idx)
                buffers = self._load_frame_buffers(cap, frames_to_load)
                
                if len(buffers) <= self.delta_t:
                    break
                
                # 후보 위치 결정
                if self.window_count < self.initial_windows:
                    candidate_positions = all_positions
                    block_type = "전체 블록"
                else:
                    candidate_positions = self._get_neighbor_positions(selected_blocks, all_positions)
                    block_type = "이웃 블록"
                
                # 모션 처리
                frame_pairs = len(buffers) - self.delta_t
                
                for i in range(frame_pairs):
                    I1 = buffers[i]
                    I2 = buffers[i + self.delta_t]
                    
                    for pos in candidate_positions:
                        y, x = pos
                        block_h, block_w = self.block_size
                        
                        if y + block_h > I1.shape[0] or x + block_w > I1.shape[1]:
                            continue
                            
                        b1 = I1[y:y + block_h, x:x + block_w]
                        b2 = I2[y:y + block_h, x:x + block_w]
                        
                        sub_shift = self._process_block_motion(b1, b2)
                        traces[pos].append(sub_shift)
                        
                        if len(traces[pos]) > window_frames:
                            traces[pos] = traces[pos][-window_frames:]                
                
                # 메모리 정리
                del buffers
                gc.collect()
                
                frame_idx += frames_to_load
                frames_since_update += frames_to_load
                
                # 윈도우 업데이트 체크
                if frames_since_update >= window_frames:
                    frames_since_update = 0
                    self.window_count += 1
                    
                    snrs, block_signal = self._process_traces_with_progress(
                        traces, candidate_positions, b, a, window_frames, fps
                    )
                    
                    if self.window_count >= self.initial_windows:
                        selected_blocks = self._select_blocks_by_cumulative_threshold(snrs)
                        selection_method = f"누적 가중치 기준 ({len(selected_blocks)}개)"
                    else:
                        sorted_snrs = sorted(snrs, key=lambda x: x[1], reverse=True)[:10]
                        selected_blocks = [pos for pos, snr in sorted_snrs if snr > 0]
                        selection_method = f"상위 10개 선택 ({len(selected_blocks)}개)"
                
                current_snrs, current_block_signal = self._process_traces(
                    traces, candidate_positions, b, a, window_frames, fps
                )
                
                blocks_to_use = selected_blocks if selected_blocks else candidate_positions[:10]
                if blocks_to_use:
                    weighted_avg = self._combine_signals(blocks_to_use, current_block_signal, current_snrs)
                    combined_signal.append(weighted_avg)
            
            result = np.concatenate(combined_signal) if combined_signal else np.array([])
            self.logger.info(f"{video_path.stem} 추출 완료 - 총 윈도우: {self.window_count}, 신호 길이: {len(result)}")
            
            return result
            
        finally:
            # 리소스 정리
            if cap is not None:
                cap.release()
            # 메모리 정리
            if 'traces' in locals():
                del traces
            if 'combined_signal' in locals():
                del combined_signal
            gc.collect()

    def patchify(self, extracted_signal: np.ndarray, patch_size: int, step: int) -> List[np.ndarray]:
        """
        추출된 신호를 윈도우 크기로 패치화
        
        Args:
            extracted_signal: 추출된 호흡 신호 (1차원 numpy array)
        
        Returns:
            np.ndarray: 패치화된 신호 (shape: (num_patches, window_size))
        """
        if len(extracted_signal) < patch_size:
            raise ValueError("추출된 신호가 윈도우 크기보다 작습니다.")
        
        patches = []
        for start in range(0, len(extracted_signal) - patch_size + 1, step):
            if start + patch_size >= len(extracted_signal):
                patches.append(extracted_signal[start:])
                break
            patches.append(extracted_signal[start:start + patch_size])
        return np.stack(patches)
    
    def save_to_npy(self, signal: np.ndarray, file_path: str) -> None:
        """
        추출된 신호를 .npy 파일로 저장
        
        Args:
            signal: 추출된 호흡 신호 (1차원 numpy array)
            file_path: 저장할 파일 경로
        """
        np.save(file_path, signal)
        self.logger.info(f"저장 완료: {file_path}")
        
import sys

# 전역 변수로 executor 관리
executor = None

def cleanup_resources():
    """리소스 정리 함수"""
    global executor
    
    print("리소스 정리 중...")
    
    # ProcessPoolExecutor 강제 종료
    if executor is not None:
        print("ProcessPoolExecutor 종료 중...")
        executor.shutdown(wait=False)
        
        # 자식 프로세스들 강제 종료
        try:
            current_process = psutil.Process()
            children = current_process.children(recursive=True)
            for child in children:
                try:
                    child.terminate()
                except psutil.NoSuchProcess:
                    pass
            
            # 강제 종료 대기
            psutil.wait_procs(children, timeout=3)
            
            # 여전히 살아있는 프로세스들 강제 킬
            for child in children:
                try:
                    if child.is_running():
                        child.kill()
                except psutil.NoSuchProcess:
                    pass
        except Exception as e:
            print(f"자식 프로세스 정리 중 오류: {e}")
    
    # 메모리 정리
    gc.collect()
    print("리소스 정리 완료")

def signal_handler(sig, frame):
    """개선된 시그널 핸들러"""
    print(f"\n시그널 {sig} 수신됨. 프로그램 종료 중...")
    cleanup_resources()
    sys.exit(0)

def process_single_video(video_path):
    """단일 비디오 처리 함수 (메모리 정리 개선)"""
    local_processor = None
    try:
        # 새로운 프로세서 인스턴스 생성
        local_processor = RespiratorSignalProcessor(
            block_size=(24, 60),
            overlap=12,
            delta_t=1,
            window_sec=30,
            cumulative_threshold=0.3,
            initial_windows=2
        )
        
        print(f"Processing video: {video_path.stem}")
        respiratory_signal = local_processor.extract_respiratory_signal(video_path)
        respiratory_signal = local_processor.patchify(respiratory_signal, patch_size=150, step=150)
        
        output_path = out_path / (video_path.stem + '.npy')
        local_processor.save_to_npy(respiratory_signal, str(output_path))
        local_processor.logger.info(f"{video_path.stem} 처리 완료")
        
        return f"Successfully processed: {video_path.stem}"
        
    except Exception as e:
        # 상세한 에러 정보 수집
        error_traceback = traceback.format_exc()
        error_line = traceback.extract_tb(e.__traceback__)[-1].lineno
        error_function = traceback.extract_tb(e.__traceback__)[-1].name
        
        error_msg = (f"Error processing {video_path.stem}:\n"
                    f"  - 에러 타입: {type(e).__name__}\n"
                    f"  - 에러 메시지: {str(e)}\n"
                    f"  - 발생 함수: {error_function}\n"
                    f"  - 발생 줄 번호: {error_line}\n"
                    f"  - 전체 Traceback:\n{error_traceback}")
        
        if local_processor:
            local_processor.logger.error(error_msg)
        else:
            print(error_msg)
        
        return f"Error processing {video_path.stem}: Line {error_line} in {error_function} - {str(e)}"
    
    finally:
        # 리소스 정리
        if local_processor:
            del local_processor
        gc.collect()

# 사용 예시
if __name__ == "__main__":
    exceptions = [
            # [1. edf 파일이 존재하지 않는 case]
            'A2019-EM-01-0119',
            'A2019-EM-01-0120',
            'A2019-EM-01-0122',
            'A2019-EM-01-0123',
            'A2019-EM-01-0124',
            'A2019-EM-01-0125',
            'A2019-EM-01-0196',
            'A2019-EM-01-0197',
            'A2019-EM-01-0198',
            'A2019-EM-01-0199',
            'A2019-EM-01-0200',
            'A2019-EM-01-0201',
            'A2019-EM-01-0202',
            'A2019-EM-01-0203',
            'A2019-EM-01-0204',
            'A2019-EM-01-0205',
            'A2019-EM-01-0206',
            # records 개수에 관한 RuntimeWarning 발생
            'A2021-EM-01-0163',
            ]
    p = Path('/tf/00_data/#_2021_Sleep_Video/')
    videos = list(p.glob('**/A*.mp4'))
    videos = [video for video in videos if video.stem[:-len('_video_01')] not in exceptions][:2]
    out_path = Path('/tf/01_code/mylittlecodes/SleepVST_baseline/data/extracted_resp_signals/test/')
    out_path.mkdir(parents=True, exist_ok=True)
    
    # 시그널 핸들러 등록 (개선됨)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    max_workers = min(len(videos), 4)
    
    with tqdm(total=len(videos), desc="전체 진행률", unit="파일") as file_progress:
        try:
            # 프로세스 풀 생성
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # 작업 제출
                future_to_video = {
                    executor.submit(process_single_video, video): video 
                    for video in videos
                }
                
                for future in concurrent.futures.as_completed(future_to_video):
                    video = future_to_video[future]
                    try:
                        result = future.result()
                        print(result)
                    except Exception as e:
                        # 메인 프로세스에서의 에러도 상세하게 로깅
                        error_traceback = traceback.format_exc()
                        error_line = traceback.extract_tb(e.__traceback__)[-1].lineno
                        error_function = traceback.extract_tb(e.__traceback__)[-1].name
                        
                        error_msg = (f"Main process error for {video.stem}:\n"
                                   f"  - 에러 타입: {type(e).__name__}\n"
                                   f"  - 에러 메시지: {str(e)}\n"
                                   f"  - 발생 함수: {error_function}\n"
                                   f"  - 발생 줄 번호: {error_line}\n"
                                   f"  - 전체 Traceback:\n{error_traceback}")
                        print(error_msg)
                    finally:
                        file_progress.update(1)
        
        except KeyboardInterrupt:
            print("Ctrl+C 감지됨. 정리 중...")
            cleanup_resources()
        
        finally:
            # 최종 정리
            cleanup_resources()
