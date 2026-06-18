"""
Respiratory signal extraction pipeline with multiprocessing support for KVSS video dataset.
This module provides a wrapper around the respiratory extraction functionality
with optional multiprocessing for video-level parallelization.
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple
from omegaconf import DictConfig
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import traceback
from src.utils.logger import get_logger
from src.data.preprocess.io import loadVideo, FrameResize, loadVideoStream, loadVideoStreamCV2
import imageio.v3 as iio
from scipy.signal import butter, sosfilt, iirnotch, lfilter
from sklearn.preprocessing import scale

from .respiratory_extraction import resp_extraction, resp_extraction_r
from .filters.temporal_filters import difference_of_iir

logger = get_logger(__name__)


def butter_bandpass(lowcut, highcut, fs, order=1):
    low = lowcut
    high = highcut
    sos  = butter(order, [low, high], btype='band', fs=fs, output='sos', analog=False)
    return sos

def butter_bandpass_filter(data, lowcut, highcut, fs, order=1):
    sos  = butter_bandpass(lowcut, highcut, fs, order=order)
    y = sosfilt(sos, data)
    return y

def notch_filter(data, f0, Q, fs):
    b, a = iirnotch(f0, Q, fs)
    y = lfilter(b, a, data)
    return y

def filtered_signal(signal, sample_freq, low=0.1, high=0.5, order=1):

#     # Notch filter to cancel out the power line disturbance (50 Hz - 60Hz)
#     f0, Q = 50, 5
#     y1 = notch_filter(signal, f0, Q, fs=sample_freq)
#     f0 = 60
#     y2 = notch_filter(y1, f0, Q, fs=sample_freq)

    #y2 = signal
    # Butterworth filter for valid freq signals
    #filtered_signal = butter_bandpass_filter(y2, low, high, fs=sample_freq, order=order)
    filtered_signal = butter_bandpass_filter(signal, low, high, fs=sample_freq, order=order)

    return filtered_signal

def normalize(y_val):
    y = (y_val - np.min(y_val)) / (np.max(y_val) - np.min(y_val))
    return y


def processing(signal):
    standardize = scale(signal)
    butter = filtered_signal(standardize, 5)
    norm = normalize(butter)
    #norm = normalize(standardize)

    return norm

def load_video(video_path: str) -> Tuple[np.ndarray, float]:
    """
    Load video file and extract frames.

    Args:
        video_path: Path to video file

    Returns:
        Tuple of (frames array, fps)
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Normalize to [0, 1]
        frame_normalized = frame_rgb.astype(np.float32) / 255.0
        frames.append(frame_normalized)

    cap.release()

    return np.array(frames), fps


def get_temporal_filter(cfg: DictConfig):
    """
    Create temporal filter function based on config.

    Args:
        cfg: Configuration object with temporal filter settings

    Returns:
        Temporal filter function
    """
    filter_type = cfg.temporal_filter.get('type', 'difference_of_iir')

    if filter_type == 'difference_of_iir':
        rl = cfg.temporal_filter.get('rl', 0.4)
        rh = cfg.temporal_filter.get('rh', 0.05)

        def temporal_filter(delta, fl, fh):
            return difference_of_iir(delta, rl, rh)

        return temporal_filter
    else:
        raise ValueError(f"Unknown temporal filter type: {filter_type}")

def parse_json(json_path):
        """
        Args:
            json_path (str)
        Returns:
            list: 수면 단계 정보가 포함된 딕셔너리 리스트
        """
        import json

        label_map = {
                "Wake": 0,
                "N1": 1,
                "N2": 1,
                "N3": 2,
                "REM": 3,
            }
        sleep_epochs = []
        with open(json_path, 'r') as f:
            ann = json.load(f)
            fps = ann["Video_Info"][0]["Frame_Rate"]

            for row in ann['Event']:
                if row['Event_Label'] not in label_map:
                    continue
                sleep_epochs.append({
                    "start": (float(row['Start_Epoch'])-1) * 30,
                    "duration": row["Duration(second)"],
                    "label": label_map[row['Event_Label']]
                })
            return fps, sleep_epochs

def get_last_epoch_fps(json_path):
    """
    Args:
        json_path (str)
    Returns:
        int: 마지막 에포크 번호
    """
    import json
    import datetime

    with open(json_path, 'r') as f:
        ann = json.load(f)
        fps = ann["Video_Info"][0]["Frame_Rate"]
        record_id = ann["Case_Info"]["Case_Number"]
        start_time = datetime.datetime.strptime(ann['Video_Info'][0]['Start'], "%Y/%m/%d %H:%M:%S.%f")
        end_time = datetime.datetime.strptime(ann['Video_Info'][0]['End'], "%Y/%m/%d %H:%M:%S.%f")
        total_epoch = (end_time - start_time).seconds // 30
        return total_epoch, fps, record_id

def process_single_video_by_epoch(
    video_path: str,
    output_dir: str,
    cfg: DictConfig,
    show_progress: bool = True
) -> str:
    """
    Process a single video for respiratory signal extraction.

    Args:
        video_path: Path to input video
        output_dir: Base output directory
        cfg: Configuration object
        show_progress: Whether to show tqdm progress bar

    Returns:
        Status message string
    """
    try:
        num_epochs, fps, record_id = get_last_epoch_fps(video_path.replace('_video_01.mp4', '_annotation.json'))
        logger.info(f"Processing video: {record_id}")

        # Check if filtering is enabled and if this record should be processed
        filter_cfg = cfg.get('filter', {})
        if filter_cfg.get('enabled', False):
            record_ids = filter_cfg.get('record_ids', [])
            if record_ids and record_id not in record_ids:
                msg = f"Skipped {record_id} (not in filter list)"
                logger.info(msg)
                return msg

        # Get epoch range for this record
        epoch_start = 0
        epoch_end = num_epochs
        if filter_cfg.get('enabled', False):
            epoch_ranges = filter_cfg.get('epoch_ranges', {})
            if record_id in epoch_ranges:
                epoch_range = epoch_ranges[record_id]
                epoch_start = epoch_range[0] - 1  # Convert to 0-indexed
                epoch_end = min(epoch_range[1], num_epochs)  # End is inclusive, but bounded by num_epochs
                logger.info(f"Processing epochs {epoch_range[0]} to {epoch_range[1]} for {record_id}")

        # Create output directory for this video
        video_output_dir = Path(output_dir) / record_id
        video_output_dir.mkdir(parents=True, exist_ok=True)

        # Open video stream once
        # reader = iio.imiter(video_path)
        # reader_iter = iter(reader)
        # frames_per_epoch = int(30 * fps)
        cap = cv2.VideoCapture(video_path)

        # FPS 일치성 확인 및 frames_per_epoch 계산을 캡처 FPS 기준으로
        cap_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        if cap_fps <= 0:
            cap_fps = fps  # fallback
        frames_per_epoch = int(round(30 * cap_fps))
        if abs(cap_fps - fps) > 0.1:
            logger.warning(f"FPS mismatch (json:{fps:.3f}, video:{cap_fps:.3f}). Using video FPS for seeking.")

        # Skip frames before epoch_start if needed
        if epoch_start > 0:
            frames_to_skip = epoch_start * frames_per_epoch
            logger.info(f"Skipping {frames_to_skip} frames to reach epoch {epoch_start + 1}")
            cap.set(cv2.CAP_PROP_POS_FRAMES, frames_to_skip)

        processed_count = 0

        preprocessed_signal_dir='data/resp_proxy_video'
        preprocessed_path = Path(preprocessed_signal_dir) / f'{record_id}_bw.npy'
        preprocessed_signal = np.load(preprocessed_path)

        # 간단한 무결성 체크 함수
        def _is_epoch_done(path: Path) -> bool:
            if not path.exists():
                return False
            try:
                arr = np.load(path, mmap_mode='r')
                return arr.size > 0
            except Exception:
                return False

        # Create iterator with or without progress bar
        epoch_range_iter = range(epoch_start, epoch_end)
        if show_progress:
            epoch_range_iter = tqdm(epoch_range_iter, mininterval=10, desc=f"Processing epochs for {record_id}")

        for i in epoch_range_iter:
            epoch_output_dir = video_output_dir / f'epoch_{i+1}'
            movement_path = epoch_output_dir / f'epoch_{i+1}_movement.npy'

            # 이미 처리된 에포크 스킵 (파일 무결성 확인 포함)
            if cfg.skip_existing and _is_epoch_done(movement_path):
                logger.info(f"Skipping epoch {i+1} - already processed")
                # 현재 위치가 목표보다 앞설 때만 시킹
                target_pos = (i + 1) * frames_per_epoch
                cur_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                if cur_pos < target_pos:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, target_pos)
                continue

            # 필요할 때만 디렉터리 생성
            epoch_output_dir.mkdir(parents=True, exist_ok=True)

            # Load frames for current epoch from stream
            epoch_video = loadVideoStreamCV2(
                cap,
                num_frames=frames_per_epoch,
                crop_box=(32, 422, 125, 515),
                size=(250, 250),
                normalize=True
            )
            preprocessed = preprocessed_signal[i]

            if epoch_video is None:
                logger.warning(f"No more frames available at epoch {i+1}")
                break

            if len(epoch_video) < frames_per_epoch:
                logger.warning(f"Epoch {i+1} has only {len(epoch_video)} frames (expected {frames_per_epoch})")

            # Extract respiratory signal for this epoch
            max_point = resp_extraction_r(
                video=epoch_video,
                fps=fps,
                mag_factor=cfg.magnification.mag_factor,
                freq_range=cfg.magnification.freq_range,
                attenuate=cfg.magnification.attenuate,
                sigma=cfg.magnification.sigma,
                temporal_filter=get_temporal_filter(cfg),
                save_dir=epoch_output_dir,
                preprocessed_signal=preprocessed,
                epoch_idx=i+1,
                record_id=record_id
            )

            processed_count += 1

        cap.release()

        if filter_cfg.get('enabled', False) and record_id in filter_cfg.get('epoch_ranges', {}):
            epoch_range = filter_cfg['epoch_ranges'][record_id]
            return f"Successfully processed epochs {epoch_range[0]}-{epoch_range[1]} ({processed_count} epochs) for {record_id}"
        else:
            return f"Successfully processed {processed_count} epochs for {record_id}"

    except Exception as e:
        error_msg = f"Error processing {Path(video_path).name}: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        return error_msg


def _process_video_wrapper(args):
    """
    Wrapper function for multiprocessing.
    Unpacks arguments and calls process_single_video_by_epoch.

    Args:
        args: Tuple of (video_path, output_dir, cfg)

    Returns:
        Status message string
    """
    video_path, output_dir, cfg = args
    # Show progress for individual videos when using multiprocessing
    return process_single_video_by_epoch(video_path, output_dir, cfg, show_progress=False)


def process_all_videos(cfg: DictConfig) -> List[str]:
    """
    Process all videos for respiratory signal extraction.
    Supports optional multiprocessing for video-level parallelization.

    Args:
        cfg: Configuration object with multiprocessing settings

    Returns:
        List of status messages for each video
    """
    results = []

    # Get input video path
    input_path = Path(cfg.video.input_path)

    if not input_path.exists():
        logger.error(f"Input path does not exist: {input_path}")
        return [f"Error: Input path not found - {input_path}"]

    # Get list of videos to process
    video_files = []

    # Check if file list is provided
    file_list_path = cfg.video.get('file_list', None)
    if file_list_path and Path(file_list_path).exists():
        logger.info(f"Loading video list from: {file_list_path}")
        with open(file_list_path, 'r') as f:
            record_ids = [line.strip() for line in f if line.strip()]

        for record_id in record_ids:
            video_path = input_path / record_id / (record_id + '_video_01.mp4')
            if video_path.exists():
                video_files.append(str(video_path))
            else:
                logger.warning(f"Video not found: {record_id}")
    else:
        # Process all videos in directory
        logger.info(f"Scanning directory for videos: {input_path}")
        video_files.extend([str(f) for f in input_path.glob('A*/*.mp4')])

    if not video_files:
        logger.error(f"No video files found in: {input_path}")
        return [f"Error: No videos found in {input_path}"]

    logger.info(f"Found {len(video_files)} videos to process")

    # Create output directory
    output_dir = Path(cfg.output.dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if multiprocessing is enabled
    use_multiprocessing = cfg.get('multiprocessing', {}).get('enabled', False)
    num_workers = cfg.get('multiprocessing', {}).get('num_workers', None)

    if use_multiprocessing:
        # Determine number of workers
        if num_workers is None or num_workers <= 0:
            num_workers = max(1, cpu_count() - 1)  # Leave one CPU free
        else:
            num_workers = min(num_workers, cpu_count())

        logger.info(f"Using multiprocessing with {num_workers} workers")

        # Prepare arguments for multiprocessing
        process_args = [(video_path, str(output_dir), cfg) for video_path in video_files]

        # Process videos in parallel
        results = []
        with Pool(processes=num_workers) as pool:
            # Use imap_unordered with chunksize=1 for immediate progress updates
            with tqdm(total=len(video_files), desc="Processing videos (parallel)", unit="video") as pbar:
                for result in pool.imap_unordered(_process_video_wrapper, process_args, chunksize=1):
                    results.append(result)
                    pbar.update(1)
                    # Optionally log each completion
                    if result.startswith("Success"):
                        pbar.set_postfix_str("✓", refresh=True)
    else:
        # Process videos sequentially (original behavior)
        logger.info("Processing videos sequentially (multiprocessing disabled)")
        results = []
        for video_path in tqdm(video_files, desc="Processing videos (sequential)", unit="video"):
            result = process_single_video_by_epoch(video_path, str(output_dir), cfg, show_progress=True)
            results.append(result)

    return results
