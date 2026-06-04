import os
from pathlib import Path
import numpy as np
import gc
import sys
import psutil
import mne
import warnings
import signal as sig
import csv
import traceback
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any

from omegaconf import DictConfig
from src.common.logger import Logger
from src.data.preprocess.io import *
from src.data.preprocess.utils_data import *
from mne.io import read_raw_edf
from tqdm import tqdm


@dataclass
class RespiratoryProcessingConfig:
    """Configuration for Respiratory EDF preprocessing"""
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
    process_hw: bool  # Whether to process heart wave
    process_bw: bool  # Whether to process breathing wave

    # Respiratory signal processing parameters
    respiratory_filter_low: float  # Low cutoff frequency (Hz)
    respiratory_filter_high: float  # High cutoff frequency (Hz)
    respiratory_filter_order: int  # Filter order
    target_fs: float  # Target sampling frequency

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
    def from_hydra_config(cls, cfg: DictConfig) -> 'RespiratoryProcessingConfig':
        """Create RespiratoryProcessingConfig from Hydra config"""
        # Get respiratory filter parameters with defaults
        resp_filter = cfg.get('respiratory_filter', {})

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
            process_hw=cfg.signals.get('process_hw', True),
            process_bw=cfg.signals.get('process_bw', True),
            respiratory_filter_low=resp_filter.get('low', 0.1),
            respiratory_filter_high=resp_filter.get('high', 0.5),
            respiratory_filter_order=resp_filter.get('order', 1),
            target_fs=resp_filter.get('target_fs', 5),
            batch_size=cfg.processing.batch_size,
            num_workers=cfg.processing.num_workers,
            timeout=cfg.processing.timeout,
            memory_threshold=cfg.processing.memory_threshold,
            skip_partial=cfg.error_handling.skip_partial,
            continue_on_error=cfg.error_handling.continue_on_error,
            max_retries=cfg.error_handling.max_retries,
            # selection (all optional)
            select_include=list(getattr(cfg, 'selection', {}).get('include', [])) if hasattr(cfg, 'selection') else [],
            select_exclude=list(getattr(cfg, 'selection', {}).get('exclude', [])) if hasattr(cfg, 'selection') else [],
            select_files=list(getattr(cfg, 'selection', {}).get('files', [])) if hasattr(cfg, 'selection') else [],
            select_file_list=getattr(cfg, 'selection', {}).get('file_list', None) if hasattr(cfg, 'selection') else None
        )


class RespiratoryEDFPreprocessor:
    """Respiratory EDF preprocessor with respiratory_extraction.py signal processing"""

    def __init__(self, config: RespiratoryProcessingConfig, logger: Logger):
        self.config = config
        self.logger = logger

        # Setup signal handlers
        sig.signal(sig.SIGINT, self._signal_handler)
        sig.signal(sig.SIGTERM, self._signal_handler)

        # Create save directory
        Path(self.config.save_dir).mkdir(parents=True, exist_ok=True)

    def _signal_handler(self, sig, frame):
        """Handle Ctrl+C and other signals"""
        self.logger.info("프로그램 종료 요청됨...")
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

    def extract_signal(self, edf_path: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Extract ECG and respiratory signals from EDF file
        Returns processed heart wave and breathing wave patches
        Uses respiratory_extraction.py signal processing for breathing wave
        Returns None for signals that are not being processed based on config
        """
        # Calculate duration from CSV annotation
        duration_sec = None
        try:
            edf_file = Path(edf_path)
            basename = edf_file.name
            ann_path = Path(self.config.annotation_dir) / basename.replace('.edf', '_label.csv')
            if ann_path.exists():
                duration_sec = int(self._get_last_row_column_value(str(ann_path), 'Start_Epoch')) * 30
                if duration_sec <= 0:
                    duration_sec = None
                    self.logger.warning(f"{basename}의 annotation 파일에서 유효한 duration을 찾을 수 없습니다.")
            else:
                self.logger.warning(f"{basename}의 annotation 파일을 찾을 수 없습니다: {ann_path}")
        except Exception as e:
            self.logger.error(f"{Path(edf_path).name}의 annotation 파일 처리 중 오류 발생: {str(e)}")

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
                channels_to_pick = [ch for ch in self.config.channels if ch in available_channels]

                if not channels_to_pick:
                    raise ValueError(f"필요한 채널을 찾을 수 없습니다: {edf_path}에서 {self.config.channels}를 찾을 수 없습니다.")

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
            except Exception:
                raise
            finally:
                mne.set_log_level(original_verbose)

        try:
            # Map channels to data indices
            ecg_idx = channel_names.index('EKG') if 'EKG' in channel_names else None
            resp_idx = channel_names.index('Thorax') if 'Thorax' in channel_names else None

            # Check required channels based on what we're processing
            if self.config.process_hw and ecg_idx is None:
                raise ValueError(f"EKG 채널이 필요하지만 찾을 수 없습니다. 발견된 채널: {channel_names}")
            if self.config.process_bw and resp_idx is None:
                raise ValueError(f"Thorax 채널이 필요하지만 찾을 수 없습니다. 발견된 채널: {channel_names}")

            hw = None
            bw = None

            # Process HW if requested
            if self.config.process_hw and ecg_idx is not None:
                ecg_signal = data[ecg_idx]
                hw = preprocess_hw(ecg_signal, fs)
                hw = patchify(hw, patch_size=self.config.hw_patch_size, step=self.config.hw_patch_step)
                del ecg_signal

            # Process BW if requested
            if self.config.process_bw and resp_idx is not None:
                resp_signal = data[resp_idx]
                bw = preprocess_bw_respiratory(
                    resp_signal, fs,
                    target_fs=self.config.target_fs,
                    low=self.config.respiratory_filter_low,
                    high=self.config.respiratory_filter_high,
                    order=self.config.respiratory_filter_order
                )
                bw = patchify(bw, patch_size=self.config.bw_patch_size, step=self.config.bw_patch_step)
                del resp_signal

            return hw, bw

        except Exception as e:
            raise type(e)(f"{str(e)} (파일: {edf_path}, 채널: {channels_to_pick})")
        finally:
            del data
            gc.collect()

    def process_file(self, edf_file: str) -> Tuple[bool, str]:
        """Process a single EDF file"""
        edf_path = Path(edf_file)
        base = edf_path.stem

        try:
            # Check which files need to be processed
            save_dir = Path(self.config.save_dir)
            hw_file = save_dir / f'{base}_hw.npy'
            bw_file = save_dir / f'{base}_bw.npy'

            hw_exists = hw_file.exists()
            bw_exists = bw_file.exists()

            # Determine if we need to process
            need_hw = self.config.process_hw and not hw_exists
            need_bw = self.config.process_bw and not bw_exists

            # If nothing needs processing, skip
            if not need_hw and not need_bw:
                return True, "skipped"

            # If skip_partial is enabled, remove existing files that need reprocessing
            if self.config.skip_partial:
                if need_hw and hw_exists:
                    hw_file.unlink()
                if need_bw and bw_exists:
                    bw_file.unlink()

            # Extract and process signals
            hw, bw = self.extract_signal(edf_file)

            # Save results (only save what was processed)
            if self.config.process_hw and hw is not None:
                np.save(str(hw_file), hw)
            if self.config.process_bw and bw is not None:
                np.save(str(bw_file), bw)

            return True, "processed"
            
        except MemoryError:
            return False, f"error_memory: 메모리 부족 - {base}"
        except Exception as e:
            return False, f"error_processing: {type(e).__name__}: {str(e)}"
        finally:
            gc.collect()

    def _get_files_to_process(self) -> List[str]:
        """Get list of EDF files to process"""
        import fnmatch

        edf_dir = Path(self.config.edf_dir)
        edf_files = list(edf_dir.glob(self.config.file_pattern))
        basenames = {f.stem: str(f) for f in edf_files}

        # Load selection from file_list if provided
        selection_from_file = []
        if self.config.select_file_list:
            list_path = Path(self.config.select_file_list)
            if list_path.exists():
                with open(list_path, 'r') as fh:
                    for line in fh:
                        name = line.strip()
                        if name:
                            selection_from_file.append(name)
            else:
                self.logger.warning(f"selection.file_list 경로가 존재하지 않습니다: {list_path}")

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
            candidate_files = [str(f) for f in edf_files]

        # Apply exclude patterns
        exclude_patterns = self.config.select_exclude or []
        if exclude_patterns:
            kept = []
            for f in candidate_files:
                base = Path(f).stem
                if any(fnmatch.fnmatch(base, pat) for pat in exclude_patterns):
                    continue
                kept.append(f)
            candidate_files = kept

        edf_files = candidate_files

        # Filter out already processed files
        files_to_process = []
        complete_count = 0
        save_dir = Path(self.config.save_dir)

        for edf_file in edf_files:
            base = Path(edf_file).stem

            # Check which files exist
            hw_file = save_dir / f'{base}_hw.npy'
            bw_file = save_dir / f'{base}_bw.npy'

            hw_exists = hw_file.exists()
            bw_exists = bw_file.exists()

            # Determine if processing is needed
            need_hw = self.config.process_hw and not hw_exists
            need_bw = self.config.process_bw and not bw_exists

            # If everything we need is already processed, count as complete
            if not need_hw and not need_bw:
                complete_count += 1
            else:
                files_to_process.append(edf_file)

        self.logger.info(f"이미 처리된 파일: {complete_count}/{len(edf_files)} ({complete_count/len(edf_files)*100:.1f}%)")
        self.logger.info(f"처리할 파일: {len(files_to_process)} 개")

        return files_to_process

    def process_dataset(self) -> Dict[str, int]:
        """Process the entire dataset"""
        # Memory information
        memory_info = self._get_memory_info()
        self.logger.info(f"사용 가능한 메모리: {memory_info['available']/1024/1024:.1f} MB / 총 메모리: {memory_info['total']/1024/1024:.1f} MB")

        self.logger.info("Respiratory EDF 전처리 작업 시작 (순차 처리)")
        self.logger.info(f"데이터셋: {self.config.dataset_name}")
        self.logger.info(f"처리 대상: HW={'O' if self.config.process_hw else 'X'}, BW={'O' if self.config.process_bw else 'X'}")
        if self.config.process_bw:
            self.logger.info(f"Respiratory filter: low={self.config.respiratory_filter_low}Hz, "
                            f"high={self.config.respiratory_filter_high}Hz, order={self.config.respiratory_filter_order}")

        # Get files to process
        files_to_process = self._get_files_to_process()

        if not files_to_process:
            self.logger.info("처리할 파일이 없습니다.")
            return {"processed": 0, "skipped": 0, "error": 0}

        # Overall results aggregation
        overall_results = {"processed": 0, "skipped": 0, "error": 0}

        # Process files sequentially with progress bar
        with tqdm(total=len(files_to_process), ncols=100, desc=f"{self.config.dataset_name}", unit="file") as pbar:
            for edf_file in files_to_process:
                file_name = Path(edf_file).name

                try:
                    success, status = self.process_file(edf_file)

                    if status.startswith("error"):
                        overall_results["error"] += 1
                        self.logger.error(f"{file_name}: {status}")
                    elif status == "skipped":
                        overall_results["skipped"] += 1
                    elif status == "processed":
                        overall_results["processed"] += 1

                    pbar.set_postfix(
                        processed=overall_results["processed"],
                        skipped=overall_results["skipped"],
                        error=overall_results["error"]
                    )

                except KeyboardInterrupt:
                    if self.config.continue_on_error:
                        self.logger.warning("처리 중단됨. 다음 파일로 진행합니다.")
                        overall_results["error"] += 1
                        continue
                    else:
                        raise

                except Exception as e:
                    overall_results["error"] += 1
                    self.logger.error(f"{file_name}: 처리 중 예상치 못한 오류\n{traceback.format_exc()}")
                    if not self.config.continue_on_error:
                        raise

                finally:
                    pbar.update(1)
                    gc.collect()

        # Log final results
        self.logger.info(f"데이터셋 {self.config.dataset_name} 처리 결과: "
                        f"처리 완료={overall_results['processed']}, "
                        f"건너뛴 파일={overall_results['skipped']}, "
                        f"오류={overall_results['error']}")

        return overall_results


def process_respiratory_edf_dataset(cfg: DictConfig, logger: Logger) -> Dict[str, int]:
    """Process EDF dataset with respiratory signal processing"""
    config = RespiratoryProcessingConfig.from_hydra_config(cfg)
    preprocessor = RespiratoryEDFPreprocessor(config, logger)
    return preprocessor.process_dataset()


def main():
    """Legacy main function - kept for backward compatibility"""
    import hydra
    from omegaconf import DictConfig

    @hydra.main(version_base=None, config_path="../../../config", config_name="defaults")
    def hydra_main(cfg: DictConfig):
        # Use respiratory EDF preprocessing configuration
        preprocess_cfg = cfg.preprocess if hasattr(cfg, 'preprocess') else cfg

        # Create logger
        log_cfg = preprocess_cfg.log if hasattr(preprocess_cfg, 'log') else {'dir': './logs', 'name': 'respiratory_edf_preprocess'}
        logger = Logger(dir=log_cfg['dir'], name=log_cfg['name'])

        # Process dataset
        results = process_respiratory_edf_dataset(preprocess_cfg, logger)

        logger.info("모든 데이터셋 처리 완료")
        logger.info(f"최종 결과: {results}")

    hydra_main()


if __name__ == "__main__":
    main()
