"""Extract optical-flow motion features from sleep videos."""

from dataclasses import dataclass
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple, Optional, Union

import cv2
import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig
from tqdm import tqdm

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class MotionProcessingConfig:
    """Hydra-instantiated settings for motion feature extraction.

    The flat field layout keeps multiprocessing workers independent from
    OmegaConf objects while ``preprocess/motionfeatures.yaml`` can still expose
    grouped, human-readable sections.
    """

    video_input_path: str
    video_file_list: Optional[Union[str, List[str]]]
    video_record_ids: Optional[Union[str, List[str]]]
    target_fps: int
    output_dir: str
    output_size: List[int]
    num_workers: int
    skip_existing: bool
    homography_source_points: List[List[float]]
    homography_scale_factor: float
    roi_person_box: List[int]
    roi_head_bottom_y: int
    motion_time_windows: List[int]
    motion_thresholds: List[float]


class MotionFeatureExtractor:
    """Compute optical-flow motion features for a single standardized video."""

    def __init__(self, config: MotionProcessingConfig):
        """Store preprocessing geometry and feature parameters."""
        self.config = config
        self.target_fps = config.target_fps
        self.out_size = tuple(config.output_size)
        self.pts_src = np.array(config.homography_source_points, dtype=np.float32)
        self.scale_factor = config.homography_scale_factor
        self.person_box = config.roi_person_box  # (x0, y0, x1, y1)
        self.head_bottom_y = config.roi_head_bottom_y
        self.time_windows = config.motion_time_windows
        self.motion_thresholds = config.motion_thresholds

    def scale_quad(self, pts: np.ndarray, scale: float) -> np.ndarray:
        """Return ``pts`` scaled around their centroid by ``scale``."""
        center = pts.mean(axis=0)
        return (pts - center) * scale + center

    def homography_transform(self, img: np.ndarray) -> np.ndarray:
        """Warp a frame into the configured bed-aligned output plane."""
        pts_dst = np.array([[0, 0], [self.out_size[0], 0], 
                           [self.out_size[0], self.out_size[1]], [0, self.out_size[1]]], 
                          dtype=np.float32)
        pts_src_scaled = self.scale_quad(self.pts_src, scale=self.scale_factor)
        
        h, _ = cv2.findHomography(pts_src_scaled, pts_dst)
        return cv2.warpPerspective(img, h, self.out_size)

    def extract_region_magnitudes_from_box(self, flow: np.ndarray) -> Dict[str, Tuple[float, float]]:
        """Summarize flow magnitude for head, body, and outer regions."""
        h, w = flow.shape[:2]
        mag = np.linalg.norm(flow, axis=2)

        # Fixed person box coordinates
        x0, y0, x1, y1 = self.person_box
        head_y1 = self.head_bottom_y

        # Create masks for different regions
        mask_H = np.zeros((h, w), dtype=np.uint8)  # Head
        mask_B = np.zeros((h, w), dtype=np.uint8)  # Body
        mask_O = np.ones((h, w), dtype=np.uint8)   # Outer

        mask_H[y0:head_y1, x0:x1] = 1
        mask_B[head_y1:y1, x0:x1] = 1
        mask_O[y0:y1, x0:x1] = 0  # Exclude person box

        def compute_stats(mask):
            vals = mag[mask == 1]
            return (np.max(vals) if vals.size > 0 else 0.0,
                   np.mean(vals) if vals.size > 0 else 0.0)

        vH, sH = compute_stats(mask_H)
        vB, sB = compute_stats(mask_B)
        vO, sO = compute_stats(mask_O)
        
        return {'Head': (vH, sH), 'Body': (vB, sB), 'Outer': (vO, sO)}

    def compute_optical_flow(self, prev_img: np.ndarray, curr_img: np.ndarray,
                             flow_calculator) -> np.ndarray:
        """Compute dense optical flow between consecutive grayscale frames."""
        return flow_calculator.calc(prev_img, curr_img, None)

    def compute_motion_features(self, v_seq: np.ndarray, s_seq: np.ndarray,
                                seconds: List[int],
                                thresholds: List[float]) -> Dict[str, np.ndarray]:
        """Build cumulative-motion and time-since-motion features."""
        T = len(v_seq)
        features = {}
        delta_fps = self.target_fps

        # Cumulative sum based f1, f2 features
        v_cum = np.cumsum(v_seq)
        s_cum = np.cumsum(s_seq)
        
        for sec in seconds:
            logger.debug(f"Computing motion features for window={sec}s")
            delta = int(sec * delta_fps)
            f1 = np.zeros(T)
            f2 = np.zeros(T)
            
            for t in range(T):
                if t >= delta:
                    f1[t] = v_cum[t] - v_cum[t - delta]
                    f2[t] = s_cum[t] - s_cum[t - delta]
                else:
                    f1[t] = v_cum[t]
                    f2[t] = s_cum[t]
                    
            features[f'f1@{sec}s'] = f1
            features[f'f2@{sec}s'] = f2

        # Time since last motion features
        for delta in thresholds:
            f3 = np.zeros(T, dtype=np.int32)
            f4 = np.zeros(T, dtype=np.int32)
            last_v = -1
            last_s = -1
            
            for t in range(T):
                if v_seq[t] > delta:
                    last_v = t
                if s_seq[t] > delta:
                    last_s = t
                    
                f3[t] = t - last_v + 1 if last_v >= 0 else t + 1
                f4[t] = t - last_s + 1 if last_s >= 0 else t + 1
                
            features[f'f3@{delta}'] = f3
            features[f'f4@{delta}'] = f4

        return features


def process_single_video(config: MotionProcessingConfig, video_info: Tuple[str, bool]) -> Optional[str]:
    """Extract motion features for one video and save them as a ``.npy`` file."""
    video_file, skip_existing = video_info

    try:
        # Check if output already exists
        video_name = Path(video_file).name
        output_filename = Path(config.output_dir) / f'{video_name.replace("_video_01.mp4", "_motion_features.npy")}'

        if skip_existing and output_filename.exists():
            logger.info(f"Skipping {video_name}, output already exists.")
            return f"Skipped: {video_name}"

        # Initialize extractor
        extractor = MotionFeatureExtractor(config)
        
        # Open video
        cap = cv2.VideoCapture(video_file)
        if not cap.isOpened():
            logger.warning(f"Could not open video file {video_file}")
            return f"Failed: {video_name} - Could not open"

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(round(fps / config.target_fps))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total_steps = max(1, int(frame_count // frame_interval) - 1)

        # Read first frame
        ret, prev = cap.read()
        if not ret:
            logger.warning(f"Could not read first frame from {video_file}")
            cap.release()
            return f"Failed: {video_name} - No frames"

        prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
        prev = extractor.homography_transform(prev)

        # Initialize optical flow calculator
        dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        
        # Initialize storage for sequences
        region_seqs = {
            'Head': {'v': [], 's': []},
            'Body': {'v': [], 's': []},
            'Outer': {'v': [], 's': []}
        }
        
        logger.info(f"Processing {video_name} with {total_steps} steps at target FPS {config.target_fps}")
        
        # Process frames
        frames_processed = 0
        for step in range(total_steps):
            # Skip frames to match target FPS
            for _ in range(frame_interval - 1):
                cap.read()
            
            ret, curr = cap.read()
            if not ret:
                break

            curr = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
            curr = extractor.homography_transform(curr)
            
            # Compute optical flow
            flow = extractor.compute_optical_flow(prev, curr, dis)
            
            # Extract region magnitudes
            region_vs = extractor.extract_region_magnitudes_from_box(flow)
            
            for region in ['Head', 'Body', 'Outer']:
                v, s = region_vs[region]
                region_seqs[region]['v'].append(v)
                region_seqs[region]['s'].append(s)

            prev = curr
            frames_processed += 1

        cap.release()
        # region_seqs length: frames_processed

        if frames_processed == 0:
            logger.warning(f"No frames processed for {video_file}")
            return f"Failed: {video_name} - No frames processed"

        # Compute features for all regions
        all_features = {}
        for region in ['Head', 'Body', 'Outer']:
            v_seq = np.array(region_seqs[region]['v'])
            s_seq = np.array(region_seqs[region]['s'])

            features = extractor.compute_motion_features(
                v_seq, s_seq,
                seconds=extractor.time_windows,
                thresholds=extractor.motion_thresholds
            )

            # 디버깅: feature 이름들 출력
            logger.debug(f"Features for {region}: {list(features.keys())}")

            for k, v in features.items():
                all_features[f'{k}_{region}'] = v

        # 디버깅: 최종 feature 목록 출력
        logger.debug(f"Final features: {sorted(all_features.keys())}")
        logger.debug(f"Total feature count: {len(all_features)}")
        
        # Save results
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        np.save(output_filename, all_features)
        
        logger.info(f"Successfully processed {video_name} ({frames_processed} frames)")
        return f"Success: {video_name} ({frames_processed} frames)"

    except Exception as e:
        logger.error(f"Error processing {video_file}: {str(e)}")
        return f"Error: {Path(video_file).name} - {str(e)}"


def _as_list(value: Optional[Union[str, List[str]]]) -> List[str]:
    """Normalize optional comma-separated strings or lists into a list."""
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(',') if item.strip()]
    return [str(item).strip() for item in value if str(item).strip()]


def get_video_files(config: MotionProcessingConfig) -> List[str]:
    """Return existing video paths selected by record IDs, file list, or scan."""
    try:
        # Priority 1: Check for record_ids in config (highest priority)
        record_ids = _as_list(config.video_record_ids)
        if record_ids:
            logger.info(f"Processing specific record IDs: {record_ids}")
            video_dir = Path(config.video_input_path)
            file_paths = [video_dir / rid / f"{rid}_video_01.mp4" for rid in record_ids]

            # Filter existing files
            existing_files = [f for f in file_paths if f.exists()]
            if len(existing_files) != len(file_paths):
                missing = set(str(f) for f in file_paths) - set(str(f) for f in existing_files)
                logger.warning(f"Missing files for record IDs: {missing}")

            file_paths = existing_files

        # Priority 2: If specific file list is provided
        elif config.video_file_list:
            if isinstance(config.video_file_list, str) and Path(config.video_file_list).exists():
                with open(config.video_file_list, 'r') as f:
                    video_dir = Path(config.video_input_path)
                    file_paths = [video_dir / line.strip() / f"{line.strip()}_video_01.mp4" for line in f if line.strip()]
            else:
                video_dir = Path(config.video_input_path)
                file_paths = [video_dir / record_id / f"{record_id}_video_01.mp4" for record_id in _as_list(config.video_file_list)]

            # Filter existing files
            existing_files = [f for f in file_paths if Path(f).exists()]
            if len(existing_files) != len(file_paths):
                missing = set(file_paths) - set(existing_files)
                logger.warning(f"Missing files: {missing}")

            file_paths = existing_files
        else:
            # Priority 3: Default - scan directory for video files
            file_paths = list(Path(config.video_input_path).glob('A*/*.mp4'))

        if not file_paths:
            logger.warning(f"No video files found")
        else:
            logger.info(f"Found {len(file_paths)} video files")

        return [str(fp) for fp in file_paths]
    except Exception as e:
        logger.error(f"Error reading video files: {e}")
        return []


def process_all_videos(config: MotionProcessingConfig) -> List[str]:
    """Process all videos selected by ``config`` using multiprocessing."""
    video_files = get_video_files(config)
    if not video_files:
        return []

    total_videos = len(video_files)

    # Prepare video info tuples
    video_infos = [(vf, config.skip_existing) for vf in video_files]

    logger.info(f"Processing {total_videos} videos with {config.num_workers} workers")

    results = []

    with ProcessPoolExecutor(max_workers=config.num_workers) as executor:
        # Submit all jobs at once
        future_to_video = {executor.submit(process_single_video, config, video_info): video_info[0]
                         for video_info in video_infos}
        
        # Collect results with progress bar
        with tqdm(total=total_videos, desc="Processing videos") as pbar:
            for future in as_completed(future_to_video):
                video_file = future_to_video[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as exc:
                    error_msg = f"Error: {Path(video_file).name} - {exc}"
                    logger.error(error_msg)
                    results.append(error_msg)
                finally:
                    pbar.update(1)
    
    return results