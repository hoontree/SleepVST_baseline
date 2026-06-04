from glob import glob
import cv2
import numpy as np
from tqdm import tqdm
import argparse
import os
import sys
import yaml
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from functools import partial
from typing import Dict, List, Tuple, Optional, Union
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Config:
    """Configuration class to handle YAML config and command line arguments."""
    
    def __init__(self, config_path: Optional[str] = None, args: Optional[argparse.Namespace] = None):
        # Default configuration
        self.config = {
            'video': {
                'input_path': '/tf/00_data/#_2021_Sleep_Video/',
                'target_fps': 4,
                'file_list': None  # Optional: specific files to process
            },
            'output': {
                'dir': '/tf/01_code/mylittlecodes/SleepVST_baseline/data/motionfeatures',
                'size': [224, 224]
            },
            'processing': {
                'num_workers': 8,
                'skip_existing': False
            },
            'homography': {
                'source_points': [[100, 50], [540, 50], [540, 430], [100, 430]],  # Default points
                'scale_factor': 1.0
            },
            'roi': {
                'person_box': [50, 30, 174, 194],  # [x0, y0, x1, y1]
                'head_bottom_y': 80
            },
            'motion_features': {
                'time_windows': [5, 10, 30, 60],  # seconds
                'motion_thresholds': [0.5, 1.0, 2.0, 5.0]  # motion thresholds
            }
        }
        
        # Load YAML config if provided
        if config_path and os.path.exists(config_path):
            self.load_yaml_config(config_path)
        
        # Override with command line arguments if provided
        if args:
            self.override_with_args(args)
    
    def load_yaml_config(self, config_path: str):
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                yaml_config = yaml.safe_load(f)
            self._deep_update(self.config, yaml_config)
            logger.info(f"Loaded configuration from {config_path}")
        except Exception as e:
            logger.warning(f"Failed to load YAML config from {config_path}: {e}")
    
    def override_with_args(self, args: argparse.Namespace):
        """Override config with command line arguments."""
        if hasattr(args, 'video_path') and args.video_path:
            self.config['video']['input_path'] = args.video_path
        if hasattr(args, 'target_fps') and args.target_fps:
            self.config['video']['target_fps'] = args.target_fps
        if hasattr(args, 'file_list') and args.file_list:
            self.config['video']['file_list'] = args.file_list
        if hasattr(args, 'out_dir') and args.out_dir:
            self.config['output']['dir'] = args.out_dir
        if hasattr(args, 'out_size') and args.out_size:
            self.config['output']['size'] = args.out_size
        if hasattr(args, 'num_workers') and args.num_workers:
            self.config['processing']['num_workers'] = args.num_workers
        if hasattr(args, 'skip_existing') and args.skip_existing:
            self.config['processing']['skip_existing'] = args.skip_existing
    
    def _deep_update(self, base_dict: dict, update_dict: dict):
        """Deep update dictionary."""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def __getattr__(self, name):
        """Allow accessing config as attributes."""
        if name in self.config:
            return SimpleNamespace(**self.config[name]) if isinstance(self.config[name], dict) else self.config[name]
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

class SimpleNamespace:
    """Simple namespace for nested config access."""
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        # Convert nested dicts to SimpleNamespace
        for key, value in self.__dict__.items():
            if isinstance(value, dict):
                self.__dict__[key] = SimpleNamespace(**value)

class MotionFeatureExtractor:
    """Class to handle motion feature extraction from videos."""
    
    def __init__(self, config: Config):
        self.config = config
        self.target_fps = config.video.target_fps
        self.out_size = tuple(config.output.size)
        self.pts_src = np.array(config.homography.source_points, dtype=np.float32)
        self.scale_factor = config.homography.scale_factor
        self.person_box = config.roi.person_box  # (x0, y0, x1, y1)
        self.head_bottom_y = config.roi.head_bottom_y
        self.time_windows = config.motion_features.time_windows
        self.motion_thresholds = config.motion_features.motion_thresholds

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

def process_single_video(video_info: Tuple[str, Config, bool]) -> Optional[str]:
    """Process a single video file to extract motion features."""
    video_file, config, skip_existing = video_info
    
    try:
        # Check if output already exists
        video_name = os.path.basename(os.path.dirname(video_file))
        output_filename = os.path.join(config.output.dir, f'{video_name}_motion_features.npy')
        
        if skip_existing and os.path.exists(output_filename):
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
        frame_interval = int(round(fps / config.video.target_fps))
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
        
        logger.info(f"Processing {video_name} with {total_steps} steps at target FPS {config.video.target_fps}")
        
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

            features = extractor.compute_motion_features(
                v_seq, s_seq, 
                seconds=extractor.time_windows, 
                thresholds=extractor.motion_thresholds
            )

            for k, v in features.items():
                all_features[f'{k}_{region}'] = v

        # Save results
        os.makedirs(config.output.dir, exist_ok=True)
        np.save(output_filename, all_features)
        
        logger.info(f"Successfully processed {video_name} ({frames_processed} frames)")
        return f"Success: {video_name} ({frames_processed} frames)"

    except Exception as e:
        logger.error(f"Error processing {video_file}: {str(e)}")
        return f"Error: {os.path.basename(os.path.dirname(video_file))} - {str(e)}"

def get_video_files(config: Config) -> List[str]:
    """Get list of video files to process."""
    try:
        # If specific file list is provided
        if config.video.file_list:
            if isinstance(config.video.file_list, str):
                # If it's a file path containing list of files
                if os.path.exists(config.video.file_list):
                    with open(config.video.file_list, 'r') as f:
                        file_paths = [line.strip() for line in f if line.strip()]
                else:
                    # If it's a comma-separated string
                    file_paths = [f.strip() for f in config.video.file_list.split(',')]
            else:
                # If it's already a list
                file_paths = config.video.file_list
                
            # Filter existing files
            existing_files = [f for f in file_paths if os.path.exists(f)]
            if len(existing_files) != len(file_paths):
                missing = set(file_paths) - set(existing_files)
                logger.warning(f"Missing files: {missing}")
            
            file_paths = existing_files
        else:
            # Default: scan directory for video files
            file_paths = glob(os.path.join(config.video.input_path, 'A*', '*.mp4'))
            
        if not file_paths:
            logger.warning(f"No video files found")
        else:
            logger.info(f"Found {len(file_paths)} video files")
            
        return file_paths
    except Exception as e:
        logger.error(f"Error reading video files: {e}")
        return []

def process_all_videos(config: Config) -> List[str]:
    """Process all videos using multiprocessing."""
    video_files = get_video_files(config)
    if not video_files:
        return []
    
    total_videos = len(video_files)
    
    # Prepare video info tuples
    video_infos = [(vf, config, config.processing.skip_existing) for vf in video_files]
    
    logger.info(f"Processing {total_videos} videos with {config.processing.num_workers} workers")
    
    results = []
    
    with ProcessPoolExecutor(max_workers=config.processing.num_workers) as executor:
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

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Compute motion features from videos using multiprocessing.")
    parser.add_argument('--config', type=str, default=None,
                       help='Path to YAML configuration file')
    parser.add_argument('--video_path', type=str, default=None,
                       help='Path to the input video directory.')
    parser.add_argument('--file_list', type=str, default=None,
                       help='Specific files to process (file path or comma-separated list)')
    parser.add_argument('--target_fps', type=int, default=None,
                       help='Target FPS for optical flow computation.')
    parser.add_argument('--out_dir', type=str, default=None,
                       help='Path to the output directory.')
    parser.add_argument('--out_size', type=int, nargs=2, default=None,
                       help='Output size for homography transformation.')
    parser.add_argument('--num_workers', type=int, default=None,
                       help='Number of worker processes')
    parser.add_argument('--skip_existing', action='store_true',
                       help='Skip videos that already have output files')
    return parser.parse_args()

def create_default_config_file(config_path: str = 'motion_config.yaml'):
    """Create a default configuration file."""
    default_config = {
        'video': {
            'input_path': '/tf/00_data/#_2021_Sleep_Video/',
            'target_fps': 4,
            'file_list': None  # Can be a file path or list of video files
        },
        'output': {
            'dir': '/tf/01_code/mylittlecodes/SleepVST_baseline/data/motionfeatures',
            'size': [224, 224]
        },
        'processing': {
            'num_workers': 8,
            'skip_existing': False
        },
        'homography': {
            'source_points': [[100, 50], [540, 50], [540, 430], [100, 430]],
            'scale_factor': 1.0
        },
        'roi': {
            'person_box': [50, 30, 174, 194],  # [x0, y0, x1, y1]
            'head_bottom_y': 80
        },
        'motion_features': {
            'time_windows': [5, 10, 30, 60],  # seconds
            'motion_thresholds': [0.5, 1.0, 2.0, 5.0]  # motion thresholds
        }
    }
    
    with open(config_path, 'w') as f:
        yaml.dump(default_config, f, default_flow_style=False, indent=2)
    
    logger.info(f"Default configuration file created: {config_path}")

def run_motion_extraction(config_path: str = None, **kwargs):
    """Main function to run motion feature extraction with simple API."""
    if config_path:
        config = Config(config_path=config_path)
    else:
        config = Config()
    
    # Override with any provided kwargs
    for key, value in kwargs.items():
        if hasattr(config, key.split('.')[0]):
            # Handle nested attributes like 'video.target_fps'
            parts = key.split('.')
            obj = config
            for part in parts[:-1]:
                obj = getattr(obj, part)
            setattr(obj, parts[-1], value)
    
    # Process videos
    results = process_all_videos(config)
    
    # Summary
    successful = sum(1 for r in results if r.startswith("Success"))
    skipped = sum(1 for r in results if r.startswith("Skipped"))
    failed = sum(1 for r in results if r.startswith("Failed") or r.startswith("Error"))
    
    logger.info("=" * 50)
    logger.info("PROCESSING SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Total videos: {len(results)}")
    logger.info(f"Successful: {successful}")
    logger.info(f"Skipped: {skipped}")
    logger.info(f"Failed: {failed}")
    logger.info("=" * 50)
    
    return results

def main():
    """Main function."""
    args = parse_arguments()
    
    # Initialize configuration
    config = Config(config_path=args.config, args=args)
    
    # Set default number of workers if not specified
    if config.processing.num_workers is None:
        config.processing.num_workers = min(multiprocessing.cpu_count(), 8)
    
    logger.info(f"Starting motion feature extraction with {config.processing.num_workers} workers")
    logger.info(f"Video path: {config.video.input_path}")
    logger.info(f"Output directory: {config.output.dir}")
    logger.info(f"Target FPS: {config.video.target_fps}")
    logger.info(f"Output size: {config.output.size}")
    
    # Process videos
    results = process_all_videos(config)
    
    if not results:
        logger.error("No videos were processed. Exiting.")
        sys.exit(1)
    
    # Summary
    successful = sum(1 for r in results if r.startswith("Success"))
    skipped = sum(1 for r in results if r.startswith("Skipped"))
    failed = sum(1 for r in results if r.startswith("Failed") or r.startswith("Error"))
    
    logger.info("=" * 50)
    logger.info("PROCESSING SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Total videos: {len(results)}")
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