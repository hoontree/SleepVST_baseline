"""Preprocess respiratory EDF files into SleepVST HW/BW patch arrays."""

from pathlib import Path
import numpy as np
import gc
import sys
import mne
import warnings
import signal as sig
import csv
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

from hydra.utils import instantiate
from omegaconf import DictConfig
from src.utils.logger import get_logger
from src.data.preprocess.io import *
from src.data.preprocess.utils_data import *
from mne.io import read_raw_edf
from tqdm import tqdm

logger = get_logger(__name__)


@dataclass
class RespiratoryProcessingConfig:
    """Hydra-instantiated settings for respiratory EDF preprocessing.

    The fields are intentionally flat so Hydra can instantiate this dataclass
    directly from ``preprocess/respiratory_edf.yaml`` while the YAML itself keeps
    human-friendly grouped sections such as ``dataset`` and ``signals``.
    """
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


class RespiratoryEDFPreprocessor:
    """Extract and save heart-wave and breathing-wave patches from EDF files."""

    def __init__(self, config: RespiratoryProcessingConfig):
        """Initialize the preprocessor and ensure the output directory exists."""
        self.config = config

        # Setup signal handlers
        sig.signal(sig.SIGINT, self._signal_handler)
        sig.signal(sig.SIGTERM, self._signal_handler)

        # Create save directory
        Path(self.config.save_dir).mkdir(parents=True, exist_ok=True)

    def _signal_handler(self, sig, frame):
        """Exit cleanly when the process receives an interrupt signal."""
        logger.info("종료 요청을 받았습니다.")
        sys.exit(0)

    def _get_last_row_column_value(self, file_path: str, column_name: str) -> str:
        """Return ``column_name`` from the final data row of a CSV file.

        Annotation files can be large, so this reads backwards from the end of
        the file instead of loading the entire CSV into memory.
        """
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
        """Load one EDF file and return processed HW/BW patch arrays.

        The returned tuple is ``(hw, bw)``. Each value is ``None`` when that
        signal type is disabled in the config. When annotation duration is
        available, the raw signal is cropped before preprocessing.
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
                    logger.debug(f"{basename}: annotation duration이 유효하지 않아 전체 EDF를 사용합니다.")
            else:
                logger.debug(f"{basename}: annotation 파일이 없어 전체 EDF를 사용합니다. path={ann_path}")
        except Exception as e:
            logger.debug(f"{Path(edf_path).name}: annotation duration 계산 실패, 전체 EDF를 사용합니다. error={e}")

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
        """Process one EDF file and persist the enabled output arrays.

        Returns a ``(success, status)`` tuple where ``status`` is one of
        ``processed``, ``skipped``, or an ``error_*`` string suitable for logs.
        """
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
        """Return EDF paths that match selection rules and still need outputs."""
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
                logger.warning(f"selection.file_list를 찾을 수 없습니다: {list_path}")

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

        logger.info(
            f"EDF 선택 결과: total={len(edf_files)}, completed={complete_count}, "
            f"pending={len(files_to_process)}"
        )

        return files_to_process

    def process_dataset(self) -> Dict[str, int]:
        """Process all pending EDF files and return aggregate counts."""
        targets = []
        if self.config.process_hw:
            targets.append("HW")
        if self.config.process_bw:
            targets.append("BW")

        logger.info(
            f"Respiratory EDF 전처리 시작: dataset={self.config.dataset_name}, "
            f"targets={','.join(targets) or 'none'}"
        )
        if self.config.process_bw:
            logger.debug(
                "BW filter config: target_fs=%s, low=%sHz, high=%sHz, order=%s",
                self.config.target_fs,
                self.config.respiratory_filter_low,
                self.config.respiratory_filter_high,
                self.config.respiratory_filter_order,
            )

        # Get files to process
        files_to_process = self._get_files_to_process()

        if not files_to_process:
            logger.info("처리할 EDF 파일이 없습니다.")
            return {"processed": 0, "skipped": 0, "error": 0}

        # Overall results aggregation
        overall_results = {"processed": 0, "skipped": 0, "error": 0}

        # Process files sequentially with progress bar
        with tqdm(total=len(files_to_process), ncols=100, desc=f"{self.config.dataset_name}", unit="file") as pbar:
            for edf_file in files_to_process:
                file_name = Path(edf_file).name

                try:
                    _success, status = self.process_file(edf_file)

                    if status.startswith("error"):
                        overall_results["error"] += 1
                        logger.error(f"{file_name}: {status}")
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
                        logger.warning("처리 중단됨. 다음 파일로 진행합니다.")
                        overall_results["error"] += 1
                        continue
                    else:
                        raise

                except Exception:
                    overall_results["error"] += 1
                    logger.exception(f"{file_name}: 처리 중 예상치 못한 오류")
                    if not self.config.continue_on_error:
                        raise

                finally:
                    pbar.update(1)
                    gc.collect()

        logger.info(
            f"Respiratory EDF 전처리 완료: processed={overall_results['processed']}, "
            f"skipped={overall_results['skipped']}, errors={overall_results['error']}"
        )

        return overall_results


def process_respiratory_edf_dataset(cfg: DictConfig) -> Dict[str, int]:
    """Instantiate preprocessing config from Hydra and process the dataset."""
    config = instantiate(cfg.config)
    preprocessor = RespiratoryEDFPreprocessor(config)
    return preprocessor.process_dataset()


def main():
    """Run respiratory EDF preprocessing through the legacy Hydra entry point."""
    import hydra
    from omegaconf import DictConfig
    from src.utils.logger import setup_logging

    @hydra.main(version_base=None, config_path="../../../config", config_name="defaults")
    def hydra_main(cfg: DictConfig):
        # Use respiratory EDF preprocessing configuration
        preprocess_cfg = cfg.preprocess if hasattr(cfg, 'preprocess') else cfg

        # Configure logging
        log_cfg = preprocess_cfg.log if hasattr(preprocess_cfg, 'log') else {'dir': './logs', 'name': 'respiratory_edf_preprocess'}
        setup_logging(log_cfg['dir'], log_cfg['name'])

        # Process dataset
        results = process_respiratory_edf_dataset(preprocess_cfg)

        logger.debug(f"최종 결과: {results}")

    hydra_main()


if __name__ == "__main__":
    main()
