from glob import glob
import cv2
import numpy as np
from tqdm import tqdm
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from functools import partial
from typing import Dict, List, Tuple, Optional
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Compute motion features from videos using multiprocessing.")
    parser.add_argument('--video_path', type=str, 
                       default='/tf/00_data/#_2021_Sleep_Video/', 
                       help='Path to the input video directory.')
    parser.add_argument('--target_fps', type=int, default=4, 
                       help='Target FPS for optical flow computation.')
    parser.add_argument('--out_dir', type=str, 
                       default='/tf/01_code/mylittlecodes/SleepVST_baseline/data/motionfeatures', 
                       help='Path to the output directory.')
    parser.add_argument('--out_size', type=int, nargs=2, default=[224, 224], 
                       help='Output size for homography transformation.')
    parser.add_argument('--num_workers', type=int, default=8,
                       help='Number of worker processes (default: 8)')

    parser.add_argument('--skip_existing', action='store_true',
                       help='Skip videos that already have output files')
    return parser.parse_args()

class MotionFeatureExtractor:
    """Class to handle motion feature extraction from videos."""
    
    def __init__(self, cfg):
        self.target_fps = cfg.target_fps
        self.out_size = tuple(cfg.out_size)
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
                              thresholds: List[float]) -> Dict[str, np.ndarray]:
        """Compute motion features from velocity sequences."""
        T = len(v_seq)
        features = {}
        delta_fps = self.target_fps

        # Cumulative sum based f1, f2 features
        v_cum = np.cumsum(v_seq)
        s_cum = np.cumsum(s_seq)
        
        for sec in seconds:
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

def process_single_video(video_info: Tuple[str, str, bool, int, Tuple[int, int]]) -> Optional[str]:
    """Process a single video file to extract motion features."""
    video_file, out_dir, skip_existing, target_fps, out_size = video_info
    
    try:
        # Check if output already exists
        video_name = os.path.basename(os.path.dirname(video_file))
        output_filename = os.path.join(out_dir, f'{video_name}_motion_features.npy')
        
        if skip_existing and os.path.exists(output_filename):
            logger.info(f"Skipping {video_name}, output already exists.")
            return f"Skipped: {video_name}"

        # Initialize extractor
        extractor = MotionFeatureExtractor(target_fps=target_fps, out_size=out_size)
        
        # Open video
        cap = cv2.VideoCapture(video_file)
        if not cap.isOpened():
            logger.warning(f"Could not open video file {video_file}")
            return f"Failed: {video_name} - Could not open"

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(round(fps / target_fps))
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
        
        print(f"Processing {video_name} with {total_steps} steps at target FPS {target_fps}")
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

        if frames_processed == 0:
            logger.warning(f"No frames processed for {video_file}")
            return f"Failed: {video_name} - No frames processed"

        # Compute features for all regions
        all_features = {}
        for region in ['Head', 'Body', 'Outer']:
            v_seq = np.array(region_seqs[region]['v'])
            s_seq = np.array(region_seqs[region]['s'])

            features = extractor.compute_motion_features(v_seq, s_seq, seconds=extractor.time_windows, thresholds=extractor.motion_thresholds)

            for k, v in features.items():
                all_features[f'{k}_{region}'] = v

        # Save results
        os.makedirs(out_dir, exist_ok=True)
        np.save(output_filename, all_features)
        
        logger.info(f"Successfully processed {video_name} ({frames_processed} frames)")
        return f"Success: {video_name} ({frames_processed} frames)"

    except Exception as e:
        logger.error(f"Error processing {video_file}: {str(e)}")
        return f"Error: {os.path.basename(os.path.dirname(video_file))} - {str(e)}"

def get_video_files(video_path: str) -> List[str]:
    """Get list of video files to process."""
    try:
        file_paths = glob(os.path.join(video_path, 'A*', '*.mp4'))
        if not file_paths:
            logger.warning(f"No video files found in {video_path}")
        return file_paths
    except Exception as e:
        logger.error(f"Error reading video directory: {e}")
        return []

def process_all_videos(video_files: List[str], out_dir: str, skip_existing: bool,
                      target_fps: int, out_size: Tuple[int, int], num_workers: int) -> List[str]:
    """Process all videos at once using multiprocessing."""
    total_videos = len(video_files)
    
    # Prepare video info tuples
    video_infos = [(vf, out_dir, skip_existing, target_fps, out_size) 
                   for vf in video_files]
    
    logger.info(f"Submitting {total_videos} videos for processing with {num_workers} workers")
    
    results = []
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all jobs at once
        future_to_video = {executor.submit(process_single_video, video_info): video_info[0] 
                         for video_info in video_infos}
        
        # Collect results with progress bar
        with tqdm(total=total_videos, desc="Processing videos") as pbar:
            for future in as_completed(future_to_video):
                video_file = future_to_video[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as exc:
                    error_msg = f"Error: {os.path.basename(os.path.dirname(video_file))} - {exc}"
                    logger.error(error_msg)
                    results.append(error_msg)
                finally:
                    pbar.update(1)
    
    return results

def main():
    """Main function."""
    args = parse_arguments()
    
    # Set default number of workers
    if args.num_workers is None:
        args.num_workers = min(multiprocessing.cpu_count(), 8)  # Cap at 8 to avoid overwhelming system
    
    logger.info(f"Starting motion feature extraction with {args.num_workers} workers")
    logger.info(f"Video path: {args.video_path}")
    logger.info(f"Output directory: {args.out_dir}")
    logger.info(f"Target FPS: {args.target_fps}")
    logger.info(f"Output size: {args.out_size}")
    
    # Get video files
    video_files = get_video_files(args.video_path)
    if not video_files:
        logger.error("No video files found. Exiting.")
        sys.exit(1)
    
    logger.info(f"Found {len(video_files)} video files to process")
    
    # Process videos
    results = process_all_videos(
        video_files=video_files,
        out_dir=args.out_dir,
        skip_existing=args.skip_existing,
        target_fps=args.target_fps,
        out_size=tuple(args.out_size),
        num_workers=args.num_workers
    )
    
    # Summary
    successful = sum(1 for r in results if r.startswith("Success"))
    skipped = sum(1 for r in results if r.startswith("Skipped"))
    failed = sum(1 for r in results if r.startswith("Failed") or r.startswith("Error"))
    
    logger.info("=" * 50)
    logger.info("PROCESSING SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Total videos: {len(video_files)}")
    logger.info(f"Successful: {successful}")
    logger.info(f"Skipped: {skipped}")
    logger.info(f"Failed: {failed}")
    logger.info("=" * 50)
    
    # Log failed videos for debugging
    failed_videos = [r for r in results if r.startswith("Failed") or r.startswith("Error")]
    if failed_videos:
        logger.info("Failed videos:")
        for failed in failed_videos:
            logger.info(f"  {failed}")
    
    logger.info("Motion feature extraction completed")

if __name__ == "__main__":
    main()
    sys.exit(0)