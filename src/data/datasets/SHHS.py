"""SHHS dataset wrappers for SleepVST training and evaluation."""

from torch.utils.data import DataLoader
from src.data.base_datamodule import BaseDataset, SampleDict, BaseDataModule
from tqdm import tqdm
from typing import List

import numpy as np


class SHHS(BaseDataset):
    """Load SHHS waveform windows and sleep-stage labels.

    The dataset expects each split directory to contain an ``A-<split>_set.txt``
    file with record IDs and, for every available record, ``*_hw.npy``,
    ``*_bw.npy`` and ``*_label.npy`` files. Training/validation samples are
    fixed-length sliding windows. Test-style samples can be returned as full
    recordings by calling :meth:`load_samples` with ``is_test=True``.
    """

    def __init__(self, root, split='train', seq_len=240, exceptions=None, **kwargs):
        """Create a SHHS dataset from Hydra yaml fields.

        Args:
            root: Dataset root directory. Split files are read from
                ``root / split``.
            split: Split name. Expected values are ``train``, ``valid`` and
                ``test``.
            seq_len: Number of epochs in one training/validation sample.
            exceptions: Optional record IDs to exclude before loading samples.
            **kwargs: Extra yaml fields, usually dataloader options. They are
                accepted for Hydra compatibility and ignored here.
        """
        self.exceptions = set(exceptions or [])
        super().__init__(root=root, split=split, seq_len=seq_len, **kwargs)

    def _load_samples(self, **kwargs) -> List[SampleDict]:
        """Load samples for the active split.

        ``BaseDataset`` calls this hook during initialization. Extra yaml fields
        are intentionally ignored so dataloader metadata such as ``batch_size``
        does not leak into sample loading.
        """
        return self.load_samples()

    def _discover_ids(self):
        """Return record IDs listed for the active split.

        Split membership is defined by text files named ``A-train_set.txt``,
        ``A-valid_set.txt`` and ``A-test_set.txt`` inside ``self.data_dir``.
        File extensions are stripped so downstream path construction can append
        each modality suffix consistently.
        """
        split_files = {
            'train': 'A-train_set.txt',
            'valid': 'A-valid_set.txt',
            'test': 'A-test_set.txt',
        }
        if self.split not in split_files:
            raise ValueError(f"Unknown split: {self.split}")

        with open(self.data_dir / split_files[self.split], 'r') as f:
            record_ids = set(line.strip().replace('.h5', '') for line in f)
        return record_ids - self.exceptions

    def load_samples(self, step=10, is_test: bool = False) -> List[SampleDict]:
        """Load waveform/label arrays and convert them to sample dictionaries.

        Args:
            step: Sliding-window stride in epochs for training/validation mode.
            is_test: If ``True``, return one full-recording sample per record
                instead of sliding windows. This keeps temporal continuity for
                recording-level inference.

        Returns:
            List of ``SampleDict`` entries. Each entry contains ``x_hw``,
            ``x_bw``, ``label``, ``subject_id`` and ``start_idx``. Arrays are
            truncated to the shortest available modality/label length for that
            record.
        """
        samples = []
        if is_test:
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
                epochs = self.parse_xml(label_file)
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

            for i in range(0, T - self.seq_len + 1, step):
                samples.append({
                    "x_hw": x_hw[i:i + self.seq_len],
                    "x_bw": x_bw[i:i + self.seq_len],
                    "label": labels[i:i + self.seq_len],
                    "subject_id": record_id,
                    "start_idx": i,
                })

        return samples


class SHHSDataModule(BaseDataModule):
    """Small datamodule wrapper kept for legacy call sites.

    New training code instantiates datasets directly from Hydra configs, but this
    class is still useful for scripts that expect train/validation/test dataloader
    methods.
    """

    def __init__(self, data_dir, batch_size=32, seq_len=30):
        """Store dataloader options and the dataset root."""
        super().__init__(data_dir, batch_size)
        self.seq_len = seq_len


    def setup(self, stage=None):
        """Instantiate split datasets for the requested stage."""
        self.train_dataset = SHHS(self.data_dir, 'train', self.seq_len)
        self.valid_dataset = SHHS(self.data_dir, 'valid', self.seq_len)
        self.test_dataset = SHHS(self.data_dir, 'test', self.seq_len)

    def train_dataloader(self):
        """Return a shuffled training dataloader."""
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        """Return a deterministic validation dataloader."""
        return DataLoader(self.valid_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        """Return a deterministic test dataloader."""
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
