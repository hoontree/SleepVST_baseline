"""
Shape tracker for motion feature extraction process.

This module tracks only the shapes and dimensions throughout the motion feature
extraction pipeline, without performing actual optical flow computation or
feature calculations. Useful for debugging dimension mismatches.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class ShapeTrackingResult:
    """Container for shape tracking results."""
    record_id: str

    # Video properties
    video_path: str
    video_exists: bool
    fps: float = 0.0
    frame_count: int = 0
    frame_width: int = 0
    frame_height: int = 0

    # Processing parameters
    target_fps: int = 4
    frame_interval: int = 0

    # Calculated dimensions
    total_steps: int = 0
    frames_that_would_be_processed: int = 0

    # Sequence lengths (what would be the shape of v_seq and s_seq)
    expected_sequence_length: int = 0

    # Feature dimensions
    feature_shapes: Dict[str, Tuple[int, ...]] = field(default_factory=dict)

    # Expected output
    expected_epochs: float = 0.0
    duration_seconds: float = 0.0

    def __str__(self):
        """Pretty print the tracking result."""
        lines = [
            "=" * 80,
            f"Shape Tracking Result for {self.record_id}",
            "=" * 80,
            "",
            "VIDEO PROPERTIES:",
            f"  Path: {self.video_path}",
            f"  Exists: {self.video_exists}",
            f"  FPS: {self.fps:.2f}",
            f"  Frame Count: {self.frame_count:,}",
            f"  Frame Size: {self.frame_width} x {self.frame_height}",
            f"  Duration: {self.duration_seconds:.2f} seconds ({self.duration_seconds/60:.2f} minutes)",
            "",
            "PROCESSING PARAMETERS:",
            f"  Target FPS: {self.target_fps}",
            f"  Frame Interval: {self.frame_interval} (every {self.frame_interval}th frame)",
            "",
            "CALCULATED DIMENSIONS:",
            f"  Total Steps: {self.total_steps:,}",
            f"  Frames to Process: {self.frames_that_would_be_processed:,}",
            f"  Expected Sequence Length: {self.expected_sequence_length:,}",
            "",
            "EXPECTED OUTPUT:",
            f"  Expected Epochs (30s): {self.expected_epochs:.2f}",
            f"  Feature Length / Expected Epochs: {self.expected_sequence_length / self.expected_epochs if self.expected_epochs > 0 else 0:.6f}",
            "",
            "FEATURE SHAPES:",
        ]

        if self.feature_shapes:
            for feature_name, shape in sorted(self.feature_shapes.items()):
                lines.append(f"  {feature_name:<30} -> {shape}")
        else:
            lines.append("  (No features computed)")

        lines.append("=" * 80)

        return "\n".join(lines)


def track_video_shapes(
    record_id: str,
    video_base_path: str = "/tf/00_data/#_2021_Sleep_Video/",
    target_fps: int = 4,
    time_windows: list = None,
    motion_thresholds: list = None,
    verbose: bool = True
) -> Optional[ShapeTrackingResult]:
    """
    Track shapes throughout the motion feature extraction process.

    This function mimics the motion feature extraction pipeline but only
    tracks dimensions without performing actual computations.

    Args:
        record_id: Record ID (e.g., 'A-0038')
        video_base_path: Base path where videos are stored
        target_fps: Target FPS for downsampling
        time_windows: Time windows for cumulative features (default: [30, 300])
        motion_thresholds: Thresholds for motion detection (default: [0.01, 0.1, 1.0])
        verbose: Print detailed information

    Returns:
        ShapeTrackingResult object with all tracked dimensions
    """
    if time_windows is None:
        time_windows = [30, 300]
    if motion_thresholds is None:
        motion_thresholds = [0.01, 0.1, 1.0]

    # Initialize result
    result = ShapeTrackingResult(
        record_id=record_id,
        video_path="",
        video_exists=False,
        target_fps=target_fps
    )

    # Construct video path
    video_path = Path(video_base_path) / record_id / f"{record_id}_video_01.mp4"
    result.video_path = str(video_path)
    result.video_exists = video_path.exists()

    if not result.video_exists:
        if verbose:
            print(f"Video file not found: {video_path}")
        return result

    # Open video to get properties
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        if verbose:
            print(f"Could not open video file: {video_path}")
        return result

    # Get video properties
    result.fps = cap.get(cv2.CAP_PROP_FPS)
    result.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    result.frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    result.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    result.duration_seconds = result.frame_count / result.fps if result.fps > 0 else 0
    result.expected_epochs = result.duration_seconds / 30.0

    cap.release()

    # Calculate frame interval (same logic as motion_extractor.py)
    result.frame_interval = int(round(result.fps / target_fps))

    # Calculate total_steps (same logic as motion_extractor.py:147)
    result.total_steps = max(1, int(result.frame_count // result.frame_interval) - 1)

    # Simulate frame processing loop
    # In the actual code, this loop processes frames and builds region_seqs
    frames_processed = 0

    # This mimics the loop in motion_extractor.py:173-197
    for step in range(result.total_steps):
        # We would skip (frame_interval - 1) frames here
        # Then read 1 frame and process it
        # For each processed frame, we append to region_seqs
        frames_processed += 1

    result.frames_that_would_be_processed = frames_processed

    # The sequence length equals frames_processed (motion_extractor.py:200)
    result.expected_sequence_length = frames_processed

    # Now calculate feature shapes
    # For each region (Head, Body, Outer), we would have v_seq and s_seq
    # Both would have shape: (expected_sequence_length,)

    regions = ['Head', 'Body', 'Outer']
    T = result.expected_sequence_length  # Length of sequences

    # Time-based cumulative features (f1, f2)
    for sec in time_windows:
        for region in regions:
            # f1 and f2 both have shape (T,) - same as input sequences
            result.feature_shapes[f'f1@{sec}s_{region}'] = (T,)
            result.feature_shapes[f'f2@{sec}s_{region}'] = (T,)

    # Motion threshold features (f3, f4)
    for threshold in motion_thresholds:
        for region in regions:
            # f3 and f4 both have shape (T,) - same as input sequences
            result.feature_shapes[f'f3@{threshold}_{region}'] = (T,)
            result.feature_shapes[f'f4@{threshold}_{region}'] = (T,)

    if verbose:
        print(result)

    return result


def batch_track_shapes(
    record_ids: list,
    video_base_path: str = "/tf/00_data/#_2021_Sleep_Video/",
    target_fps: int = 4,
    verbose: bool = False
) -> Dict[str, ShapeTrackingResult]:
    """
    Track shapes for multiple records.

    Args:
        record_ids: List of record IDs
        video_base_path: Base path where videos are stored
        target_fps: Target FPS for downsampling
        verbose: Print detailed information for each record

    Returns:
        Dictionary mapping record_id to ShapeTrackingResult
    """
    results = {}

    for record_id in record_ids:
        result = track_video_shapes(
            record_id=record_id,
            video_base_path=video_base_path,
            target_fps=target_fps,
            verbose=verbose
        )
        results[record_id] = result

    return results


def compare_shapes_with_actual(
    record_id: str,
    actual_features_path: Optional[str] = None,
    video_base_path: str = "/tf/00_data/#_2021_Sleep_Video/",
    target_fps: int = 4
) -> Dict[str, any]:
    """
    Compare tracked shapes with actual extracted features.

    Args:
        record_id: Record ID
        actual_features_path: Path to actual .npy file (if None, auto-detect)
        video_base_path: Base path where videos are stored
        target_fps: Target FPS used for extraction

    Returns:
        Dictionary with comparison results
    """
    # Track expected shapes
    tracked = track_video_shapes(
        record_id=record_id,
        video_base_path=video_base_path,
        target_fps=target_fps,
        verbose=False
    )

    if not tracked.video_exists:
        return {"error": "Video not found"}

    # Try to load actual features
    if actual_features_path is None:
        # Try common locations
        possible_paths = [
            f"/tf/01_code/mylittlecodes/SleepVST_baseline/data/motionfeatures/{record_id}_motion_features.npy",
            f"/tf/01_code/mylittlecodes/SleepVST_baseline/data/motionfeatures_test/{record_id}_motion_features.npy",
        ]

        actual_features_path = None
        for path in possible_paths:
            if Path(path).exists():
                actual_features_path = path
                break

    if actual_features_path is None or not Path(actual_features_path).exists():
        return {
            "tracked": tracked,
            "actual": None,
            "comparison": "Actual features file not found"
        }

    # Load actual features
    actual_features = np.load(actual_features_path, allow_pickle=True).item()

    # Compare
    comparison = {
        "record_id": record_id,
        "expected_sequence_length": tracked.expected_sequence_length,
        "actual_feature_length": len(next(iter(actual_features.values()))),
        "match": tracked.expected_sequence_length == len(next(iter(actual_features.values()))),
        "difference": len(next(iter(actual_features.values()))) - tracked.expected_sequence_length,
        "expected_feature_count": len(tracked.feature_shapes),
        "actual_feature_count": len(actual_features),
        "expected_features": set(tracked.feature_shapes.keys()),
        "actual_features": set(actual_features.keys()),
        "missing_features": set(tracked.feature_shapes.keys()) - set(actual_features.keys()),
        "extra_features": set(actual_features.keys()) - set(tracked.feature_shapes.keys()),
    }

    return {
        "tracked": tracked,
        "actual": actual_features,
        "comparison": comparison
    }


def print_comparison(comparison: dict):
    """Pretty print comparison results."""
    if "error" in comparison:
        print(f"Error: {comparison['error']}")
        return

    if isinstance(comparison["comparison"], str):
        print(comparison["comparison"])
        return

    comp = comparison["comparison"]

    print("=" * 80)
    print(f"Shape Comparison for {comp['record_id']}")
    print("=" * 80)
    print(f"\nSEQUENCE LENGTH:")
    print(f"  Expected: {comp['expected_sequence_length']:,}")
    print(f"  Actual:   {comp['actual_feature_length']:,}")
    print(f"  Match:    {comp['match']}")
    print(f"  Difference: {comp['difference']}")

    print(f"\nFEATURE COUNT:")
    print(f"  Expected: {comp['expected_feature_count']}")
    print(f"  Actual:   {comp['actual_feature_count']}")

    if comp['missing_features']:
        print(f"\nMISSING FEATURES:")
        for feat in sorted(comp['missing_features']):
            print(f"  - {feat}")

    if comp['extra_features']:
        print(f"\nEXTRA FEATURES:")
        for feat in sorted(comp['extra_features']):
            print(f"  - {feat}")

    if comp['match'] and not comp['missing_features'] and not comp['extra_features']:
        print(f"\n✓ All shapes match perfectly!")

    print("=" * 80)
