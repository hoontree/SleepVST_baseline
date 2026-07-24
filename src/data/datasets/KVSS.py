"""KVSS dataset support for SleepVST finetuning and video transfer."""

from pathlib import Path
from typing import List

import numpy as np
import scipy.signal
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.base_datamodule import BaseDataModule, BaseDataset, SampleDict
from src.utils.logger import get_logger

logger = get_logger(__name__)


class KVSS(BaseDataset):
    """Load KVSS recordings for sensor finetuning or video-transfer experiments.

    The regular path loads ``*_hw.npy``, ``*_bw.npy`` and ``*_label.npy`` files
    from ``root``. Transfer mode additionally loads optical-flow motion features
    and, when ``data_source == 'video'``, per-epoch respiratory signals extracted
    from video frames.
    """

    def __init__(
        self,
        root,
        split='train',
        seq_len=240,
        motion_dir=None,
        signal_dir=None,
        label_dir=None,
        video_dir=None,
        respiratory_signal_dir=None,
        model='SleepVST',
        mode='finetune',
        data_source='video',
        exceptions=None,
        bw_patch_samples=74,
        **kwargs,
    ):
        """Create a KVSS dataset from Hydra yaml fields.

        Args:
            root: Directory containing split definition files and regular signal
                arrays.
            split: Dataset split to load. Expected values are ``train``,
                ``valid`` and ``test``.
            seq_len: Sliding-window length for regular non-test samples.
            motion_dir: Directory containing ``*_motion_features.npy`` files for
                transfer mode.
            signal_dir: Directory containing raw ``*_hw.npy`` and ``*_bw.npy``
                signals. Defaults to ``root``.
            label_dir: Directory containing CSV labels for raw-signal transfer
                mode.
            video_dir: Directory containing per-recording annotation JSON files.
            respiratory_signal_dir: Directory containing video-derived
                respiratory ``epoch_*/*.npy`` files.
            model: Model name. Breath-only models load BW transfer samples;
                other models load both HW and BW.
            mode: Runtime mode. ``transfer`` enables motion-feature loading;
                other values use regular signal/label loading.
            data_source: ``video`` to use extracted respiratory signals, or
                another value to use raw BW arrays from ``signal_dir``.
            exceptions: Optional record IDs to exclude before sample loading.
            bw_patch_samples: Expected sample count of one video-derived BW
                epoch file. Epochs that don't match are resampled (or, for a
                short trailing epoch, dropped) so every recording stacks into a
                uniform-length array. Defaults to 74, the v1 (frame-halved)
                proxy convention; pass 150 for v2 proxy output, which is
                already captured at the native 5 Hz BW rate.
            **kwargs: Extra yaml fields such as dataloader options. They are
                accepted for Hydra compatibility and ignored here.
        """
        logger.info(
            "KVSS dataset | split %s | root %s | motion_dir %s | signal_dir %s | label_dir %s",
            split,
            root,
            motion_dir,
            signal_dir,
            label_dir,
        )

        self.motion_dir = Path(motion_dir) if motion_dir is not None else None
        self.signal_dir = Path(signal_dir) if signal_dir is not None else Path(root)
        self.label_dir = Path(label_dir) if label_dir is not None else Path(root)
        self.video_dir = Path(video_dir) if video_dir is not None else Path(root)
        self.respiratory_signal_dir = (
            Path(respiratory_signal_dir) if respiratory_signal_dir is not None else Path(root)
        )
        self.model = model
        self.mode = mode
        self.data_source = data_source
        self.motion_feature_keys = None
        self.exceptions = set(exceptions or [])
        self.bw_patch_samples = bw_patch_samples

        super().__init__(root=root, split=split, seq_len=seq_len, **kwargs)

    def _setup(self, **kwargs):
        """Prepare directories used by KVSS sample loading.

        ``BaseDataset`` sets ``self.data_dir`` to ``root / split`` by default,
        but KVSS signal arrays live directly under ``root``. This hook restores
        that dataset-specific path convention before samples are loaded.
        """
        self.data_dir = self.root
        self.signal_dir.mkdir(parents=True, exist_ok=True)
        if self.mode == 'transfer' and self.motion_dir is not None:
            self.motion_dir.mkdir(parents=True, exist_ok=True)

    def _load_samples(self, **kwargs) -> List[SampleDict]:
        """Dispatch to the loader required by the active runtime mode.

        Returns:
            A list of sample dictionaries. Transfer samples include a ``motion``
            matrix; regular samples include only waveform arrays and labels.
        """
        if self.mode == 'transfer':
            if 'BW' in self.model:
                logger.info("Loading motion features for BW model")
                return self.load_motion_samples_bw()
            logger.info("Loading motion features for general model")
            return self.load_motion_samples()
        return self.load_regular_samples()

    def _discover_ids(self):
        """Return record IDs listed for the active split.

        Split membership is defined by ``A-train_set.txt``, ``A-valid_set.txt``
        and ``A-test_set.txt`` in ``root``. IDs listed in ``exceptions`` are
        removed before loading.
        """
        split_to_file = {
            'train': 'A-train_set.txt',
            'valid': 'A-valid_set.txt',
            'test': 'A-test_set.txt',
        }
        if self.split not in split_to_file:
            raise ValueError(f"Unknown split: {self.split}")

        list_file = self.root / split_to_file[self.split]
        with open(list_file, 'r') as f:
            original_list = set(line.strip().replace('.h5', '') for line in f)

        logger.debug("Exceptions: %s", sorted(self.exceptions))
        excluded = original_list & self.exceptions
        discovered = original_list - self.exceptions

        if excluded:
            logger.info("[%s] Excluded %d samples due to exceptions", self.split.upper(), len(excluded))
            logger.debug("[%s] Excluded sample IDs: %s", self.split.upper(), sorted(excluded))

        logger.info(
            "[%s] Total samples: %d -> %d (excluded: %d)",
            self.split.upper(),
            len(original_list),
            len(discovered),
            len(excluded),
        )

        if self.split == 'train':
            self.train_list = discovered
        elif self.split == 'valid':
            self.valid_list = discovered
        else:
            self.test_list = discovered

        return discovered

    def load_regular_samples(self) -> List[SampleDict]:
        """Load raw KVSS signal samples without video motion features.

        Test split samples preserve whole recordings for sequence-level
        inference. Train/validation samples are ``seq_len`` windows with a fixed
        stride of 10 epochs.
        """
        samples = []
        if self.split == 'test':
            for record_id in tqdm(self._discover_ids()):
                if record_id in self.exceptions:
                    continue

                hw_file = self.data_dir / f"{record_id}_hw.npy"
                bw_file = self.data_dir / f"{record_id}_bw.npy"
                label_file = self.data_dir / f"{record_id}_label.npy"

                if not hw_file.exists() or not bw_file.exists() or not label_file.exists():
                    continue

                x_hw = np.load(hw_file).astype(np.float32)
                x_bw = np.load(bw_file).astype(np.float32)
                epochs = BaseDataset.parse_csv(label_file)
                labels = np.array([e['label'] for e in epochs], dtype=np.int64)
                T = min(len(x_hw), len(x_bw), len(labels))

                samples.append({
                    "x_hw": x_hw[:T],
                    "x_bw": x_bw[:T],
                    "label": labels[:T],
                    "subject_id": record_id,
                    "start_idx": 0,
                })

            return samples

        for record_id in tqdm(self._discover_ids()):
            if record_id in self.exceptions:
                continue

            hw_file = self.data_dir / f"{record_id}_hw.npy"
            bw_file = self.data_dir / f"{record_id}_bw.npy"
            label_file = self.data_dir / f"{record_id}_label.npy"

            if not hw_file.exists() or not bw_file.exists() or not label_file.exists():
                continue

            x_hw = np.load(hw_file).astype(np.float32)
            x_bw = np.load(bw_file).astype(np.float32)
            epochs = self.parse_csv(label_file)
            labels = np.array([e['label'] for e in epochs], dtype=np.int64)
            T = min(len(x_hw), len(x_bw), len(labels))

            for i in range(0, T - self.seq_len + 1, 10):
                samples.append({
                    "x_hw": x_hw[i:i + self.seq_len],
                    "x_bw": x_bw[i:i + self.seq_len],
                    "label": labels[i:i + self.seq_len],
                    "subject_id": record_id,
                    "start_idx": i,
                })

        return samples

    def _create_temporal_features(self, feature_list, motion_keys=None):
        """Expand per-epoch motion features with local temporal context.

        Args:
            feature_list: Motion tensor of shape ``(N, patch_size, F)`` where
                ``N`` is the number of epochs and ``F`` is the number of motion
                feature channels.
            motion_keys: Optional names for the ``F`` channels. When provided,
                ``past_``, ``current_`` and ``future_`` names are cached for RF
                feature-importance reports.

        Returns:
            Array of shape ``(N, 3 * F)`` containing previous, current and next
            motion summaries. Edge epochs are padded with the nearest available
            value so the output length stays aligned with labels.
        """
        N, _, _ = feature_list.shape
        feature_list = np.mean(feature_list, axis=1)
        padded_features = np.pad(feature_list, ((3, 3), (0, 0)), mode='edge')

        past_features = padded_features[0:N]
        current_features = padded_features[3:N + 3]
        future_features = padded_features[6:N + 6]
        expanded = np.concatenate([past_features, current_features, future_features], axis=1)

        if motion_keys is not None and self.motion_feature_keys is None:
            self.motion_feature_keys = [
                f"{prefix}_{key}"
                for prefix in ['past', 'current', 'future']
                for key in motion_keys
            ]

        return expanded

    def load_motion_samples_bw(self) -> List[SampleDict]:
        """Load transfer samples for breath-only models.

        Each returned sample contains full-recording BW features, temporally
        expanded motion features, labels and subject metadata. BW signals come
        from video-derived respiratory epochs when ``data_source == 'video'``;
        otherwise raw ``*_bw.npy`` files are used.
        """
        samples = []
        discovered_ids = self._discover_ids()
        logger.info("Discovered %d IDs for split %s", len(discovered_ids), self.split)

        for record_id in tqdm(discovered_ids):
            motion_file = self.motion_dir / f"{record_id}_motion_features.npy"
            label_json = self.video_dir / record_id / f"{record_id}_annotation.json"
            fps, epochs = BaseDataset.parse_json(label_json)

            x_bw_list = []
            if self.data_source == 'video':
                root = self.respiratory_signal_dir / record_id
                signal_files = list(root.glob('epoch_*/*.npy'))

                def extract_num(path: Path):
                    """Sort epoch directories by their numeric suffix."""
                    return int(path.parent.name.split('_')[1])

                signal_files = sorted(signal_files, key=extract_num)
                if not all([signal_files, label_json.exists(), motion_file.exists()]):
                    continue
                if signal_files:
                    last = signal_files[-1]
                    arr = np.load(last).astype(np.float32)
                    if arr.shape[0] != self.bw_patch_samples:
                        signal_files = signal_files[:-1]
                for signal_file in signal_files:
                    x_bw_part = np.load(signal_file).astype(np.float32)
                    if x_bw_part.shape[0] != self.bw_patch_samples:
                        x_bw_part = scipy.signal.resample_poly(x_bw_part, self.bw_patch_samples, x_bw_part.shape[0])
                    x_bw_list.append(x_bw_part)
                x_bw = np.vstack(x_bw_list)
            else:
                signal_file = self.signal_dir / f"{record_id}_bw.npy"
                if not all([signal_file.exists(), label_json.exists(), motion_file.exists()]):
                    logger.warning("Missing files for %s, skipping", record_id)
                    continue
                x_bw = np.load(signal_file).astype(np.float32)

            motion = np.load(motion_file, allow_pickle=True).item()
            motion_features = []
            motion_keys = sorted(motion.keys())
            for key in motion_keys:
                patches = self.patchify(motion[key], int(fps * 30))
                motion_features.append(patches)

            expanded_motion_features = self._create_temporal_features(
                np.array(motion_features).transpose(1, 2, 0),
                motion_keys=motion_keys,
            )
            labels = np.array([e['label'] for e in epochs], dtype=np.int64)
            T = min(len(x_bw), len(expanded_motion_features), len(labels))

            samples.append({
                "x_bw": x_bw[:T],
                "motion": expanded_motion_features[:T],
                "label": labels[:T],
                "subject_id": record_id,
                "start_idx": 0,
            })

        logger.info("Total samples loaded: %d", len(samples))
        return samples

    def load_motion_samples(self) -> List[SampleDict]:
        """Load transfer samples for dual-input models.

        This follows the same motion/label alignment as
        :meth:`load_motion_samples_bw`, but also includes ``x_hw`` loaded from
        raw signal files for models that consume both heart and breath inputs.
        """
        samples = []
        discovered_ids = self._discover_ids()
        logger.info("Discovered %d IDs for split %s", len(discovered_ids), self.split)

        for record_id in tqdm(discovered_ids):
            hw_signal_file = self.signal_dir / f"{record_id}_hw.npy"
            motion_file = self.motion_dir / f"{record_id}_motion_features.npy"
            label_json = self.video_dir / record_id / f"{record_id}_annotation.json"
            fps, epochs = BaseDataset.parse_json(label_json)

            x_hw = np.load(hw_signal_file).astype(np.float32)
            x_bw_list = []
            if self.data_source == 'video':
                root = self.respiratory_signal_dir / record_id
                signal_files = list(root.glob('epoch_*/*.npy'))

                def extract_num(path: Path):
                    """Sort epoch directories by their numeric suffix."""
                    return int(path.parent.name.split('_')[1])

                signal_files = sorted(signal_files, key=extract_num)
                if not all([signal_files, label_json.exists(), motion_file.exists()]):
                    continue
                if signal_files:
                    last = signal_files[-1]
                    arr = np.load(last).astype(np.float32)
                    if arr.shape[0] != self.bw_patch_samples:
                        signal_files = signal_files[:-1]
                for signal_file in signal_files:
                    x_bw_part = np.load(signal_file).astype(np.float32)
                    if x_bw_part.shape[0] != self.bw_patch_samples:
                        x_bw_part = scipy.signal.resample_poly(x_bw_part, self.bw_patch_samples, x_bw_part.shape[0])
                    x_bw_list.append(x_bw_part)
                x_bw = np.vstack(x_bw_list)
            else:
                signal_file = self.signal_dir / f"{record_id}_bw.npy"
                if not all([signal_file.exists(), label_json.exists(), motion_file.exists()]):
                    logger.warning("Missing files for %s, skipping", record_id)
                    continue
                x_bw = np.load(signal_file).astype(np.float32)

            motion = np.load(motion_file, allow_pickle=True).item()
            motion_features = []
            motion_keys = sorted(motion.keys())
            for key in motion_keys:
                patches = self.patchify(motion[key], patch_size=int(fps * 30))
                motion_features.append(patches)

            motion_features = np.array(motion_features)
            expanded_motion_features = self._create_temporal_features(
                motion_features.transpose(1, 2, 0),
                motion_keys=motion_keys,
            )
            labels = np.array([e['label'] for e in epochs], dtype=np.int64)
            T = min(len(x_bw), len(motion_features[0]), len(labels))

            samples.append({
                "x_bw": x_bw[:T],
                "x_hw": x_hw[:T],
                "motion": expanded_motion_features[:T],
                "label": labels[:T],
                "subject_id": record_id,
                "start_idx": 0,
            })

        logger.info("Total samples loaded: %d", len(samples))
        return samples

    def get_motion_feature_keys(self):
        """Return RF feature names for temporally expanded motion columns."""
        return self.motion_feature_keys

    def __getitem__(self, idx):
        """Return the raw sample dictionary.

        KVSS transfer samples include a ``motion`` field, which the base dataset
        tensor-conversion helper does not know about. PyTorch's default collate
        function converts numpy arrays to tensors for the dataloader.
        """
        return self.samples[idx]

    def normalize(self, patch):
        """Return a zero-mean, unit-variance version of one signal patch."""
        return (patch - np.mean(patch)) / (np.std(patch) + 1e-6)

    def patchify(self, signal, patch_size):
        """Split a 1D signal into non-overlapping patches.

        Args:
            signal: One-dimensional signal array.
            patch_size: Number of samples per patch.

        Returns:
            Array of shape ``(num_patches, patch_size)``.
        """
        patches = []
        for start in range(0, len(signal) - patch_size + 1, patch_size):
            patches.append(signal[start:start + patch_size])
        return np.stack(patches)

    def patch_mean(self, signal, patch_size):
        """Return the mean value of each non-overlapping signal patch."""
        patches = self.patchify(signal, patch_size)
        return np.mean(patches, axis=1)


class KVSSDataModule(BaseDataModule):
    """Dataloader wrapper for an already-instantiated KVSS dataset."""

    def __init__(self, dataset, batch_size=1, num_workers=8, shuffle=True, pin_memory=True, **kwargs):
        """Store dataloader options.

        Args:
            dataset: KVSS dataset instance.
            batch_size: Batch size for the returned dataloader.
            num_workers: Worker processes used by ``DataLoader``.
            shuffle: Whether the dataloader shuffles samples.
            pin_memory: Whether CUDA pinned-memory transfer is enabled.
            **kwargs: Extra yaml fields accepted for Hydra compatibility.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.pin_memory = pin_memory

    def get_dataloader(self):
        """Build a dataloader and fail early if the dataset is empty."""
        if len(self.dataset) == 0:
            raise ValueError(f"Dataset is empty for split '{self.dataset.split}'. "
                             f"Check data paths and file existence.")

        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=self.shuffle,
            pin_memory=self.pin_memory,
        )
