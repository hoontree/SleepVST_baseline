import cv2
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple, Optional
from pathlib import Path

class MotionFeatureExtractor:
    """Class to handle motion feature extraction from videos."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.target_fps = cfg.target_fps
        self.out_size = tuple(cfg.output.size)
        self.pts_src = np.array(cfg.homography.source_points, dtype=np.float32)
        self.scale_factor = cfg.homography.scale_factor
        self.person_box = cfg.roi.person_box  # (x0, y0, x1, y1)
        self.head_bottom_y = cfg.roi.head_bottom_y
        self.time_windows = cfg.motion_features.time_windows
        self.motion_thresholds = cfg.motion_features.motion_thresholds

    def scale_quad(self, pts: np.ndarray, scale: float) -> np.ndarray:
        """Scale a quadrilateral around its centroid."""
        center = pts.mean(axis=0)
        return (pts - center) * scale + center  

    def homography_transform(self, img: np.ndarray) -> np.ndarray:
        """Apply homography transformation to standardize the image."""
        pts_dst = np.array([[0, 0], [self.out_size[0], 0], 
                           [self.out_size[0], self.out_size[1]], [0, self.out_size[1]]], 
                          dtype=np.float32)
        pts_src_scaled = self.scale_quad(self.pts_src, scale=self.scale_factor)
        
        h, _ = cv2.findHomography(pts_src_scaled, pts_dst)
        return cv2.warpPerspective(img, h, self.out_size)

    def extract_region_magnitudes_from_box(self, flow: np.ndarray) -> Dict[str, Tuple[float, float]]:
        """Extract motion magnitudes from predefined regions."""
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
        """Compute optical flow between two images."""
        return flow_calculator.calc(prev_img, curr_img, None)

    def compute_motion_features(self, v_seq: np.ndarray, s_seq: np.ndarray, 
                              seconds: List[int], 
                              thresholds: List[float], logger) -> Dict[str, np.ndarray]:
        """Compute motion features from velocity sequences."""
        T = len(v_seq)
        features = {}
        delta_fps = self.target_fps

        # Cumulative sum based f1, f2 features
        v_cum = np.cumsum(v_seq)
        s_cum = np.cumsum(s_seq)
        
        for sec in seconds:
            logger.info(f"Computing features for time window: {sec}s")
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

def process_single_video(cfg, video_info: Tuple[str, bool], logger) -> Optional[str]:
    """Process a single video file to extract motion features."""
    video_file, skip_existing = video_info

    try:
        # Check if output already exists
        video_name = Path(video_file).name
        output_filename = Path(cfg.output.dir) / f'{video_name.replace("_video_01.mp4", "_motion_features.npy")}'

        if skip_existing and output_filename.exists():
            logger.info(f"Skipping {video_name}, output already exists.")
            return f"Skipped: {video_name}"

        # Initialize extractor
        extractor = MotionFeatureExtractor(cfg)
        
        # Open video
        cap = cv2.VideoCapture(video_file)
        if not cap.isOpened():
            logger.warning(f"Could not open video file {video_file}")
            return f"Failed: {video_name} - Could not open"

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(round(fps / cfg.target_fps))
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
        
        logger.info(f"Processing {video_name} with {total_steps} steps at target FPS {cfg.target_fps}")
        
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
                thresholds=extractor.motion_thresholds,
                logger=logger
            )

            # 디버깅: feature 이름들 출력
            logger.info(f"Features for {region}: {list(features.keys())}")

            for k, v in features.items():
                all_features[f'{k}_{region}'] = v

        # 디버깅: 최종 feature 목록 출력
        logger.info(f"Final features: {sorted(all_features.keys())}")
        logger.info(f"Total feature count: {len(all_features)}")
        
        # Save results
        Path(cfg.output.dir).mkdir(parents=True, exist_ok=True)
        np.save(output_filename, all_features)
        
        logger.info(f"Successfully processed {video_name} ({frames_processed} frames)")
        return f"Success: {video_name} ({frames_processed} frames)"

    except Exception as e:
        logger.error(f"Error processing {video_file}: {str(e)}")
        return f"Error: {Path(video_file).name} - {str(e)}"

def get_video_files(cfg, logger) -> List[str]:
    """Get list of video files to process."""
    try:
        # Priority 1: Check for record_ids in config (highest priority)
        if cfg.video.get('record_ids'):
            record_ids = cfg.video.record_ids
            if isinstance(record_ids, str):
                # Handle comma-separated string
                record_ids = [rid.strip() for rid in record_ids.split(',') if rid.strip()]

            logger.info(f"Processing specific record IDs: {record_ids}")
            video_dir = Path(cfg.video.input_path)
            file_paths = [video_dir / rid / f"{rid}_video_01.mp4" for rid in record_ids]

            # Filter existing files
            existing_files = [f for f in file_paths if f.exists()]
            if len(existing_files) != len(file_paths):
                missing = set(str(f) for f in file_paths) - set(str(f) for f in existing_files)
                logger.warning(f"Missing files for record IDs: {missing}")

            file_paths = existing_files

        # Priority 2: If specific file list is provided
        elif cfg.video.get('file_list'):
            if isinstance(cfg.video.file_list, str):
                # If it's a file path containing list of files
                if Path(cfg.video.file_list).exists():
                    with open(cfg.video.file_list, 'r') as f:
                        video_dir = Path(cfg.video.input_path)
                        file_paths = [video_dir / line.strip() / f"{line.strip()}_video_01.mp4" for line in f if line.strip()]
                else:
                    # If it's a comma-separated string
                    video_dir = Path(cfg.video.input_path)
                    file_paths = [video_dir / f.strip() / f"{f.strip()}_video_01.mp4" for f in cfg.video.file_list.split(',')]
            else:
                # If it's already a list
                file_paths = cfg.video.file_list

            # Filter existing files
            existing_files = [f for f in file_paths if Path(f).exists()]
            if len(existing_files) != len(file_paths):
                missing = set(file_paths) - set(existing_files)
                logger.warning(f"Missing files: {missing}")

            file_paths = existing_files
        else:
            # Priority 3: Default - scan directory for video files
            file_paths = list(Path(cfg.video.input_path).glob('A*/*.mp4'))

        if not file_paths:
            logger.warning(f"No video files found")
        else:
            logger.info(f"Found {len(file_paths)} video files")

        return [str(fp) for fp in file_paths]
    except Exception as e:
        logger.error(f"Error reading video files: {e}")
        return []

def process_all_videos(cfg, logger) -> List[str]:
    """Process all videos using multiprocessing."""
    video_files = get_video_files(cfg, logger)
    if not video_files:
        return []
    
    total_videos = len(video_files)
    
    # Prepare video info tuples
    video_infos = [(vf, cfg.skip_existing) for vf in video_files]
    
    logger.info(f"Processing {total_videos} videos with {cfg.num_workers} workers")
    
    results = []

    with ProcessPoolExecutor(max_workers=cfg.num_workers) as executor:
        # Submit all jobs at once
        future_to_video = {executor.submit(process_single_video, cfg, video_info, logger): video_info[0] 
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