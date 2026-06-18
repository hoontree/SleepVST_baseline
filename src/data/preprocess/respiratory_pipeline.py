"""
Respiratory signal extraction pipeline for KVSS video dataset.
This module provides a wrapper around the respiratory extraction functionality
to integrate with the config-based CLI system.
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple
from omegaconf import DictConfig
from tqdm import tqdm
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
        
def get_last_epoch_fps_new(json_path):
    """
    Args:
        json_path (str)
    Returns:
        int: 마지막 에포크 번호
    """
    import json
    import datetime
    sleep_stages = ['Wake', 'N1', 'N2', 'N3', 'REM']

    with open(json_path, 'r') as f:
        ann = json.load(f)
        events = ann["Event"]
        last_epoch = 0
        for event in events:
            if event["Event_Label"] in sleep_stages:
                epoch_num = event["Start_Epoch"]
                if epoch_num > last_epoch:
                    last_epoch = epoch_num
    
    return last_epoch, ann["Video_Info"][0]["Frame_Rate"], ann["Case_Info"]["Case_Number"]

def process_single_video_by_epoch(
    video_path: str,
    output_dir: str,
    cfg: DictConfig
) -> str:
    """
    Process a single video for respiratory signal extraction.

    Args:
        video_path: Path to input video
        output_dir: Base output directory
        cfg: Configuration object

    Returns:
        Status message string
    """
    video_name = Path(video_path).stem
    logger.info(f"Processing video: {video_name}")

    num_epochs, fps, record_id = get_last_epoch_fps(video_path.replace('_video_01.mp4', '_annotation.json'))

    # Check if filtering is enabled and if this record should be processed
    filter_cfg = cfg.get('filter', {})
    if filter_cfg.get('enabled', False):
        record_ids = filter_cfg.get('record_ids', [])
        if record_ids and record_id not in record_ids:
            logger.info(f"Skipping record {record_id} - not in filter list")
            return f"Skipped {record_id} (not in filter list)"

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
    reader = iio.imiter(video_path)
    reader_iter = iter(reader)
    frames_per_epoch = int(30 * fps)
    cap = cv2.VideoCapture(video_path)

    # Skip frames before epoch_start if needed
    if epoch_start > 0:
        frames_to_skip = epoch_start * frames_per_epoch
        logger.info(f"Skipping {frames_to_skip} frames to reach epoch {epoch_start + 1}")
        # Set video position to start epoch
        cap.set(cv2.CAP_PROP_POS_FRAMES, frames_to_skip)

    processed_count = 0
    
    preprocessed_signal_dir='data/kiss_respiratory'
    preprocessed_path = Path(preprocessed_signal_dir) / f'{record_id}_bw.npy'
    preprocessed_signal = np.load(preprocessed_path)
    for i in tqdm(range(epoch_start, epoch_end), mininterval=10, desc=f"Processing epochs for {video_name}"):
        # Check if already processed
        movement_path = video_output_dir / f'epoch_{i+1}_movement.npy'
        mag_movement_path = video_output_dir / f'epoch_{i+1}_magnified_movement.npy'
        if cfg.skip_existing and movement_path.exists() and mag_movement_path.exists():
            logger.info(f"Skipping epoch {i+1} - already processed")
            # Skip frames for this epoch
            cap.set(cv2.CAP_PROP_POS_FRAMES, (i + 1) * frames_per_epoch)
            continue
        
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
        
        epoch_output_dir = video_output_dir / f'epoch_{i+1}'
        epoch_output_dir.mkdir(parents=True, exist_ok=True)

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
            epoch_idx=i+1
        )
        
        # logger.info(f"Epoch {i+1}/{num_epochs} processed. Max point: {max_point}")
        processed_count += 1

    cap.release()

    if filter_cfg.get('enabled', False) and record_id in filter_cfg.get('epoch_ranges', {}):
        epoch_range = filter_cfg['epoch_ranges'][record_id]
        return f"Successfully processed epochs {epoch_range[0]}-{epoch_range[1]} ({processed_count} epochs) for {record_id}"
    else:
        return f"Successfully processed {processed_count} epochs for {record_id}"

def process_all_videos(cfg: DictConfig) -> List[str]:
    """
    Process all videos for respiratory signal extraction.

    Args:
        cfg: Configuration object

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
            video_names = [line.strip() for line in f if line.strip()]

        for video_name in video_names:
            video_path = input_path / video_name / (video_name + '_video_01.mp4')
            if video_path.exists():
                video_files.append(str(video_path))
                break
            else:
                # Check if video_name already has extension
                video_path = input_path / video_name / (video_name + '_video_01.mp4')
                if video_path.exists():
                    video_files.append(str(video_path))
                else:
                    logger.warning(f"Video not found: {video_name}")
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

    # Process each video
    for video_path in tqdm(video_files, desc="Processing videos", mininterval=10):
        result = process_single_video_by_epoch(video_path, str(output_dir), cfg)
        results.append(result)

    return results
