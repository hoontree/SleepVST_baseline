"""
Helper functions for testing motion feature extraction on specific records.

This module provides convenient functions to extract motion features from
individual records for debugging and testing purposes.
"""

import numpy as np
from pathlib import Path
from omegaconf import OmegaConf
from src.common.logger import Logger
from src.data.preprocess.motion_extractor import process_single_video, MotionFeatureExtractor
import cv2


def extract_motion_features_from_record(
    record_id: str,
    video_base_path: str = "/tf/00_data/#_2021_Sleep_Video/",
    output_dir: str = "/tf/01_code/mylittlecodes/SleepVST_baseline/data/motionfeatures_test",
    config_path: str = "/tf/01_code/mylittlecodes/SleepVST_baseline/config/preprocess/motionfeatures.yaml",
    skip_existing: bool = False,
    verbose: bool = True
):
    """
    Extract motion features from a specific record.

    Args:
        record_id: The record ID (e.g., 'A-0038')
        video_base_path: Base path where video files are stored
        output_dir: Directory to save output features
        config_path: Path to motion features config file
        skip_existing: Whether to skip if output already exists
        verbose: Whether to print detailed logs

    Returns:
        dict: Extracted motion features, or None if failed

    Example:
        >>> features = extract_motion_features_from_record('A-0038')
        >>> print(features.keys())
        >>> print(features['f1@30s_Head'].shape)
    """
    # Load config
    cfg = OmegaConf.load(config_path)
    cfg.video.input_path = video_base_path
    cfg.output.dir = output_dir
    cfg.skip_existing = skip_existing

    # Setup logger
    logger = Logger(cfg.log.dir, name=f"motion_test_{record_id}")

    if verbose:
        logger.info(f"Extracting motion features for record: {record_id}")

    # Construct video path
    video_file = Path(video_base_path) / record_id / f"{record_id}_video_01.mp4"

    if not video_file.exists():
        logger.error(f"Video file not found: {video_file}")
        return None

    # Process the video
    video_info = (str(video_file), skip_existing)
    result = process_single_video(cfg, video_info, logger)

    if verbose:
        logger.info(f"Result: {result}")

    # Load and return the generated features
    output_file = Path(output_dir) / f"{record_id}_motion_features.npy"
    if output_file.exists():
        features = np.load(output_file, allow_pickle=True).item()
        if verbose:
            logger.info(f"Loaded features from: {output_file}")
            logger.info(f"Feature keys: {list(features.keys())}")
            logger.info(f"Feature shape (example): {next(iter(features.values())).shape}")
        return features
    else:
        logger.error(f"Output file not found: {output_file}")
        return None


def get_video_info(record_id: str, video_base_path: str = "/tf/00_data/#_2021_Sleep_Video/"):
    """
    Get basic information about a video file.

    Args:
        record_id: The record ID (e.g., 'A-0038')
        video_base_path: Base path where video files are stored

    Returns:
        dict: Video information (fps, frame_count, duration, etc.)
    """
    video_file = Path(video_base_path) / record_id / f"{record_id}_video_01.mp4"

    if not video_file.exists():
        print(f"Video file not found: {video_file}")
        return None

    cap = cv2.VideoCapture(str(video_file))
    if not cap.isOpened():
        print(f"Could not open video file: {video_file}")
        return None

    info = {
        'record_id': record_id,
        'path': str(video_file),
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }

    info['duration_seconds'] = info['frame_count'] / info['fps']
    info['duration_minutes'] = info['duration_seconds'] / 60
    info['expected_epochs'] = info['duration_seconds'] / 30  # 30-second epochs

    # Calculate expected motion feature length at target_fps=4
    target_fps = 4
    frame_interval = int(round(info['fps'] / target_fps))
    info['expected_motion_steps'] = max(1, int(info['frame_count'] // frame_interval) - 1)

    cap.release()

    return info


def compare_motion_features(record_id: str, features: dict = None):
    """
    Compare extracted motion features with expected values based on video properties.

    Args:
        record_id: The record ID
        features: Extracted features dict (if None, will try to load from default location)

    Returns:
        dict: Comparison results
    """
    # Get video info
    video_info = get_video_info(record_id)
    if not video_info:
        return None

    # Load features if not provided
    if features is None:
        output_dir = "/tf/01_code/mylittlecodes/SleepVST_baseline/data/motionfeatures_test"
        output_file = Path(output_dir) / f"{record_id}_motion_features.npy"
        if output_file.exists():
            features = np.load(output_file, allow_pickle=True).item()
        else:
            print(f"Features not found at: {output_file}")
            return None

    # Get feature length
    feature_length = len(next(iter(features.values())))

    comparison = {
        'record_id': record_id,
        'video_info': video_info,
        'feature_length': feature_length,
        'expected_motion_steps': video_info['expected_motion_steps'],
        'expected_epochs': video_info['expected_epochs'],
        'feature_length_to_epochs': feature_length / video_info['expected_epochs'],
        'difference': feature_length - video_info['expected_motion_steps'],
    }

    # Print summary
    print(f"\n{'='*60}")
    print(f"Motion Feature Analysis for {record_id}")
    print(f"{'='*60}")
    print(f"Video FPS: {video_info['fps']:.2f}")
    print(f"Total frames: {video_info['frame_count']}")
    print(f"Duration: {video_info['duration_minutes']:.2f} minutes ({video_info['duration_seconds']:.2f} seconds)")
    print(f"Expected epochs (30s): {video_info['expected_epochs']:.2f}")
    print(f"Expected motion steps (target_fps=4): {video_info['expected_motion_steps']}")
    print(f"Actual feature length: {feature_length}")
    print(f"Difference: {comparison['difference']}")
    print(f"Feature length / Expected epochs: {comparison['feature_length_to_epochs']:.6f}")
    print(f"{'='*60}\n")

    return comparison


def batch_extract_records(
    record_ids: list,
    video_base_path: str = "/tf/00_data/#_2021_Sleep_Video/",
    output_dir: str = "/tf/01_code/mylittlecodes/SleepVST_baseline/data/motionfeatures_test",
    skip_existing: bool = False
):
    """
    Extract motion features from multiple records.

    Args:
        record_ids: List of record IDs
        video_base_path: Base path where video files are stored
        output_dir: Directory to save output features
        skip_existing: Whether to skip if output already exists

    Returns:
        dict: Results for each record
    """
    results = {}

    for record_id in record_ids:
        print(f"\nProcessing {record_id}...")
        features = extract_motion_features_from_record(
            record_id=record_id,
            video_base_path=video_base_path,
            output_dir=output_dir,
            skip_existing=skip_existing,
            verbose=True
        )

        if features is not None:
            results[record_id] = {
                'success': True,
                'features': features,
                'feature_count': len(features),
                'feature_length': len(next(iter(features.values())))
            }
        else:
            results[record_id] = {
                'success': False,
                'error': 'Failed to extract features'
            }

    # Print summary
    print(f"\n{'='*60}")
    print(f"Batch Processing Summary")
    print(f"{'='*60}")
    successful = sum(1 for r in results.values() if r['success'])
    print(f"Total records: {len(record_ids)}")
    print(f"Successful: {successful}")
    print(f"Failed: {len(record_ids) - successful}")
    print(f"{'='*60}\n")

    return results
