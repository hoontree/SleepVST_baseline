import os
import os.path as path
import numpy as np
import gc
import sys
import psutil
import multiprocessing
import mne
import concurrent.futures
import warnings
import signal as sig
import csv
import traceback
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any

from concurrent.futures import ProcessPoolExecutor
from omegaconf import DictConfig
from src.utils.logger import get_logger
from src.data.preprocess.io import *
from src.data.preprocess.utils_data import *
from mne.io import read_raw_edf
from glob import glob
from tqdm import tqdm

logger = get_logger(__name__)


@dataclass
class ProcessingConfig:
    """Configuration for SNUH preprocessing"""
    # Dataset paths
    dataset_name: str
    edf_dir: str
    annotation_dir: str
    save_dir: str
    file_pattern: str
    
    # Signal processing
    channels: List[str]
    hw_patch_size: int
    bw_patch_size: int
    hw_patch_step: int
    bw_patch_step: int
    
    # Processing parameters
    batch_size: int
    num_workers: int
    timeout: int
    memory_threshold: float
    
    # Error handling
    skip_partial: bool
    continue_on_error: bool
    max_retries: int

    # Selection
    select_include: List[str]
    select_exclude: List[str]
    select_files: List[str]
    select_file_list: Optional[str]

    @classmethod
    def from_hydra_config(cls, cfg: DictConfig) -> 'ProcessingConfig':
        """Create ProcessingConfig from Hydra config"""
        return cls(
            dataset_name=cfg.dataset.name,
            edf_dir=cfg.dataset.edf_dir,
            annotation_dir=cfg.dataset.annotation_dir,
            save_dir=cfg.dataset.save_dir,
            file_pattern=cfg.dataset.file_pattern,
            channels=cfg.signals.channels,
            hw_patch_size=cfg.signals.patch_sizes.hw,
            bw_patch_size=cfg.signals.patch_sizes.bw,
            hw_patch_step=cfg.signals.patch_steps.hw,
            bw_patch_step=cfg.signals.patch_steps.bw,
            batch_size=cfg.processing.batch_size,
            num_workers=cfg.processing.num_workers,
            timeout=cfg.processing.timeout,
            memory_threshold=cfg.processing.memory_threshold,
            skip_partial=cfg.error_handling.skip_partial,
            continue_on_error=cfg.error_handling.continue_on_error,
            max_retries=cfg.error_handling.max_retries
            ,
            # selection (all optional)
            select_include=list(getattr(cfg, 'selection', {}).get('include', [])) if hasattr(cfg, 'selection') else [],
            select_exclude=list(getattr(cfg, 'selection', {}).get('exclude', [])) if hasattr(cfg, 'selection') else [],
            select_files=list(getattr(cfg, 'selection', {}).get('files', [])) if hasattr(cfg, 'selection') else [],
            select_file_list=getattr(cfg, 'selection', {}).get('file_list', None) if hasattr(cfg, 'selection') else None
        )


class SNUHPreprocessor:
    """SNUH dataset preprocessor with configuration support"""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.executor = None
        
        # Setup signal handlers
        sig.signal(sig.SIGINT, self._signal_handler)
        sig.signal(sig.SIGTERM, self._signal_handler)
        
        # Create save directory
        os.makedirs(self.config.save_dir, exist_ok=True)

    def _signal_handler(self, sig, frame):
        """Handle Ctrl+C and other signals"""
        logger.info("프로그램 종료 요청됨. 실행 중인 작업 종료 중...")
        if self.executor:
            self.executor.shutdown(wait=False)
        sys.exit(0)

    def _get_memory_info(self) -> Dict[str, float]:
        """Get system memory usage information"""
        memory = psutil.virtual_memory()
        return {
            'total': memory.total,
            'available': memory.available,
            'used': memory.used,
            'percent': memory.percent
        }

    def _get_last_row_column_value(self, file_path: str, column_name: str) -> str:
        """Get the last row value from a specific column in CSV file"""
        with open(file_path, 'r', newline='') as f:
            f.seek(0, 2)  # Move to end of file
            file_size = f.tell()

            buffer = bytearray()
            pointer = file_size - 1

            # Read line backwards
            while pointer >= 0:
                f.seek(pointer)
                byte = f.read(1)
                if byte == '\n' and buffer:
                    break
                buffer.insert(0, ord(byte))
                pointer -= 1

            last_line = buffer.decode('utf-8')

            # Read header
            f.seek(0)
            reader = csv.reader(f)
            header = next(reader)
            col_index = header.index(column_name)

            # Parse last line
            last_values = list(csv.reader([last_line]))[0]
            return last_values[col_index]
    
    def extract_signal(self, edf_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract ECG and respiratory signals from EDF file
        Returns processed heart wave and breathing wave patches
        """
        # Calculate duration from CSV annotation
        try:
            basename = os.path.basename(edf_path)
            ann_path = path.join(self.config.annotation_dir, basename.replace('.edf', '_label.csv'))
            if os.path.exists(ann_path):
                duration_sec = int(self._get_last_row_column_value(ann_path, 'Start_Epoch')) * 30
                if duration_sec <= 0:
                    duration_sec = None
                    logger.warning(f"Warning: {basename}의 XML 파일에서 유효한 duration을 찾을 수 없습니다.")
            else:
                duration_sec = None
                logger.warning(f"Warning: {basename}의 XML 파일을 찾을 수 없습니다: {ann_path}")
        except Exception as e:
            duration_sec = None
            logger.error(f"Error: {os.path.basename(edf_path)}의 XML 파일 처리 중 오류 발생: {str(e)}")
        
        # Configure MNE to minimize output
        original_verbose = mne.set_log_level('ERROR')
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                # Load EDF with minimal verbosity
                raw = read_raw_edf(edf_path, preload=False, verbose=False)
                
                # Check available channels
                available_channels = raw.ch_names
                
                # Find required channels
                channels_to_pick = []
                channel_map = {}
                
                for i, ch in enumerate(self.config.channels):
                    if ch in available_channels:
                        channels_to_pick.append(ch)
                        channel_map[ch] = i
                
                if len(channels_to_pick) < 1:
                    error_msg = f"필요한 채널을 찾을 수 없습니다: {edf_path}에서 {self.config.channels}를 찾을 수 없습니다."
                    raise ValueError(error_msg)
                
                # Select required channels only
                raw.pick(channels_to_pick)
                channel_names = raw.ch_names
                
                # Load data
                raw.load_data(verbose=False)
                
                # Crop to specified duration if available
                if duration_sec is not None and duration_sec > 0:
                    actual_duration = raw.times[-1]
                    if duration_sec < actual_duration:
                        raw.crop(tmin=0, tmax=duration_sec, include_tmax=False)
                
                data, _ = raw[:]
                fs = raw.info['sfreq']
                
                # Clean up raw object
                del raw
            except Exception as e:
                raise
            finally:
                mne.set_log_level(original_verbose)
        
        try:
            if len(data) < 2:
                raise ValueError(f"데이터셋에는 EKG와 Thorax 두 채널이 필요합니다. 발견된 채널: {channels_to_pick}")
            
            # Map channels to data indices
            ecg_idx = channel_names.index('EKG') if 'EKG' in channel_names else None
            resp_idx = channel_names.index('Thorax') if 'Thorax' in channel_names else None
            
            if ecg_idx is None or resp_idx is None:
                raise ValueError(f"필요한 채널이 없습니다. 발견된 채널: {channel_names}")
            
            ecg_signal = data[ecg_idx]  # EKG
            resp_signal = data[resp_idx]  # Thorax
            
            # Memory cleanup
            del data
            
            # Preprocess signals
            hw = preprocess_hw(ecg_signal, fs)
            del ecg_signal
            
            bw = preprocess_bw(resp_signal, fs)
            del resp_signal
            
            # Create patches
            hw = patchify(hw, patch_size=self.config.hw_patch_size, step=self.config.hw_patch_step)
            bw = patchify(bw, patch_size=self.config.bw_patch_size, step=self.config.bw_patch_step)
            
            return hw, bw
        
        except Exception as e:
            raise type(e)(f"{str(e)} (파일: {edf_path}, 채널: {channels_to_pick})")

    def process_file(self, edf_file: str) -> Tuple[bool, str]:
        """Process a single EDF file"""
        base = os.path.splitext(os.path.basename(edf_file))[0]

        # Ignore keyboard interrupt and system signals in child processes
        sig.signal(sig.SIGINT, sig.SIG_IGN)
        sig.signal(sig.SIGTERM, sig.SIG_IGN)

        try:
            processed, status = check_file_processed(base, self.config.save_dir)
            if processed:
                return True, "skipped"
            elif status == "partial" and self.config.skip_partial:
                # Remove partially processed files
                try:
                    hw_file = os.path.join(self.config.save_dir, base + '_hw.npy')
                    bw_file = os.path.join(self.config.save_dir, base + '_bw.npy')
                    if os.path.exists(hw_file):
                        os.remove(hw_file)
                    if os.path.exists(bw_file):
                        os.remove(bw_file)
                except Exception as e:
                    return False, f"error_removing_partial: {str(e)}"

            # Memory optimization
            gc.collect()
            
            # Extract and process signals
            hw, bw = self.extract_signal(edf_file)

            # Save results
            np.save(os.path.join(self.config.save_dir, base + '_hw.npy'), hw)
            np.save(os.path.join(self.config.save_dir, base + '_bw.npy'), bw)

            # Memory cleanup
            del hw, bw
            gc.collect()

            return True, "processed"
        except MemoryError:
            return False, f"error_memory: 메모리 부족 - {base}"
        except Exception as e:
            return False, f"error_processing: {type(e).__name__}: {str(e)}"

    def _get_files_to_process(self) -> List[str]:
        """Get list of EDF files to process"""
        import fnmatch

        edf_files = glob(os.path.join(self.config.edf_dir, self.config.file_pattern))
        basenames = {path.splitext(path.basename(f))[0]: f for f in edf_files}

        # Load selection from file_list if provided
        selection_from_file = []
        if self.config.select_file_list:
            list_path = self.config.select_file_list
            if os.path.exists(list_path):
                with open(list_path, 'r') as fh:
                    for line in fh:
                        name = line.strip()
                        if name:
                            selection_from_file.append(name)
            else:
                logger.warning(f"selection.file_list 경로가 존재하지 않습니다: {list_path}")

        # Compose include patterns
        include_patterns = []
        include_patterns.extend(self.config.select_include or [])
        include_patterns.extend(self.config.select_files or [])
        include_patterns.extend(selection_from_file)

        # If include_patterns is provided, filter to those only; otherwise start with all
        if include_patterns:
            selected_keys = set()
            for pat in include_patterns:
                # Allow either exact names or glob-style patterns
                matched = [k for k in basenames.keys() if fnmatch.fnmatch(k, pat)]
                if not matched and pat in basenames:
                    matched = [pat]
                selected_keys.update(matched)
            candidate_files = [basenames[k] for k in sorted(selected_keys) if k in basenames]
        else:
            candidate_files = list(edf_files)

        # Apply exclude patterns
        exclude_patterns = self.config.select_exclude or []
        if exclude_patterns:
            kept = []
            for f in candidate_files:
                base = path.splitext(path.basename(f))[0]
                if any(fnmatch.fnmatch(base, pat) for pat in exclude_patterns):
                    continue
                kept.append(f)
            candidate_files = kept
        
        edf_files = candidate_files
        
        # Filter out already processed files
        files_to_process = []
        complete_count = 0
        
        for edf_file in edf_files:
            base = path.splitext(path.basename(edf_file))[0]
            processed, _ = check_file_processed(base, self.config.save_dir)
            if processed:
                complete_count += 1
            else:
                files_to_process.append(edf_file)
        
        logger.info(f"이미 처리된 파일: {complete_count}/{len(edf_files)} ({complete_count/len(edf_files)*100:.1f}%)")
        logger.info(f"처리할 파일: {len(files_to_process)} 개")
        
        return files_to_process

    def process_dataset(self) -> Dict[str, int]:
        """Process the entire dataset"""
        # Memory information
        memory_info = self._get_memory_info()
        logger.info(f"사용 가능한 메모리: {memory_info['available']/1024/1024:.1f} MB / 총 메모리: {memory_info['total']/1024/1024:.1f} MB")

        logger.info("전처리 작업 시작")
        logger.info(f"데이터셋: {self.config.dataset_name}")
        logger.info(f"설정: batch_size={self.config.batch_size}, num_workers={self.config.num_workers}")

        # Get files to process
        files_to_process = self._get_files_to_process()
        
        if not files_to_process:
            logger.info("처리할 파일이 없습니다.")
            return {"processed": 0, "skipped": 0, "error": 0}

        # Overall results aggregation
        overall_results = {"processed": 0, "skipped": 0, "error": 0}
        
        # Process in batches
        batches = [files_to_process[i:i + self.config.batch_size] 
                  for i in range(0, len(files_to_process), self.config.batch_size)]
        
        logger.info(f"배치 크기: {self.config.batch_size}, 워커 수: {self.config.num_workers}")

        # Process batches with progress bar
        with tqdm(total=len(files_to_process), ncols=100, desc=f"{self.config.dataset_name}", unit="file") as pbar:
            for batch_idx, batch in enumerate(batches):
                try:
                    # Initialize worker pool (create new for each batch)
                    try:
                        multiprocessing.set_start_method('spawn', force=True)
                    except RuntimeError:
                        pass  # Already set
                    
                    with ProcessPoolExecutor(max_workers=self.config.num_workers) as executor_local:
                        self.executor = executor_local
                        
                        # Submit batch jobs
                        futures = {
                            executor_local.submit(self.process_file, edf_file): path.basename(edf_file) 
                            for edf_file in batch
                        }
                        
                        # Batch results aggregation
                        batch_results = {"processed": 0, "skipped": 0, "error": 0}
                        
                        # Process results with timeout
                        for future in concurrent.futures.as_completed(futures):
                            file_name = futures[future]
                            pbar.update(1)
                            
                            try:
                                success, status = future.result(timeout=self.config.timeout)
                                
                                if status.startswith("error"):
                                    batch_results["error"] += 1
                                    logger.error(f"{file_name}: {status}")
                                elif status == "skipped":
                                    batch_results["skipped"] += 1
                                elif status == "processed":
                                    batch_results["processed"] += 1
                                    logger.debug(f"Processed: {file_name}")
                                
                                pbar.set_postfix(
                                    processed=overall_results["processed"] + batch_results["processed"], 
                                    skipped=overall_results["skipped"] + batch_results["skipped"], 
                                    error=overall_results["error"] + batch_results["error"]
                                )
                            
                            except concurrent.futures.TimeoutError:
                                batch_results["error"] += 1
                                logger.error(f"{file_name} 처리 시간 초과 ({self.config.timeout}초)")
                            
                            except Exception as e:
                                batch_results["error"] += 1
                                logger.error(f"{file_name} 처리 중 예상치 못한 오류: {traceback.format_exc()}")
                    
                    # Aggregate results
                    overall_results["processed"] += batch_results["processed"]
                    overall_results["skipped"] += batch_results["skipped"]
                    overall_results["error"] += batch_results["error"]
                    
                    # Explicit garbage collection after batch
                    gc.collect()
                    
                except KeyboardInterrupt:
                    if self.config.continue_on_error:
                        logger.warning("배치 처리 중단됨. 다음 배치로 진행합니다.")
                        continue
                    else:
                        raise
                        
                except Exception as e:
                    error_msg = f"배치 {batch_idx+1}/{len(batches)} 처리 중 오류 발생: {str(e)}"
                    logger.error(error_msg)
                    if not self.config.continue_on_error:
                        raise
                    continue
                
                finally:
                    self.executor = None
        
        # Log final results
        summary = (f"데이터셋 {self.config.dataset_name} 처리 결과: "
                  f"처리 완료={overall_results['processed']}, "
                  f"건너뛴 파일={overall_results['skipped']}, "
                  f"오류={overall_results['error']}")
        
        logger.info(summary)
        return overall_results


def process_snuh_dataset(cfg: DictConfig) -> Dict[str, int]:
    """Process SNUH dataset with configuration"""
    config = ProcessingConfig.from_hydra_config(cfg)
    preprocessor = SNUHPreprocessor(config)
    return preprocessor.process_dataset()

def main():
    """Legacy main function - kept for backward compatibility"""
    import hydra
    from omegaconf import DictConfig
    from src.utils.logger import setup_logging

    @hydra.main(version_base=None, config_path="../../../config", config_name="defaults")
    def hydra_main(cfg: DictConfig):
        # Use SNUH preprocessing configuration
        preprocess_cfg = cfg.preprocess if hasattr(cfg, 'preprocess') else cfg

        # Configure logging
        log_cfg = preprocess_cfg.log if hasattr(preprocess_cfg, 'log') else {'dir': './logs', 'name': 'snuh_preprocess'}
        setup_logging(log_cfg['dir'], log_cfg['name'])

        # Process dataset
        results = process_snuh_dataset(preprocess_cfg)

        logger.info("모든 데이터셋 처리 완료")
        logger.info(f"최종 결과: {results}")

    hydra_main()


if __name__ == "__main__":
    main()