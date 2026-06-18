"""Shared dataset and dataloader utilities for SleepVST data pipelines."""

from __future__ import annotations

import csv
import datetime
import json
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, TypedDict, Union

import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset


class SampleDict(TypedDict, total=False):
    """Common sample schema used by SleepVST datasets.

    Keys:
        x_hw: Heart-waveform array or tensor of shape ``(T, H)``.
        x_bw: Breath-waveform array or tensor of shape ``(T, B)``.
        label: Integer sleep-stage labels of shape ``(T,)``.
        subject_id: Recording or subject identifier.
        start_idx: Start epoch of the sample within the source recording.
    """

    x_hw: Union[np.ndarray, torch.Tensor]
    x_bw: Union[np.ndarray, torch.Tensor]
    label: Union[np.ndarray, torch.Tensor]
    subject_id: str
    start_idx: int


class BaseDataset(Dataset, ABC):
    """Base class for waveform datasets with split-aware sample loading.

    Subclasses provide split discovery and sample loading through
    :meth:`_discover_ids` and :meth:`_load_samples`. The base initializer stores
    common metadata, runs subclass setup, and eagerly builds ``self.samples`` so
    downstream training loops can use standard PyTorch ``Dataset`` semantics.
    """

    seq_len: int
    split: str
    data_dir: Path

    def __init__(self, root, split, seq_len, **kwargs):
        """Initialize common dataset state and load samples.

        Args:
            root: Dataset root directory.
            split: Active split name, usually ``train``, ``valid`` or ``test``.
            seq_len: Number of epochs expected in fixed-length samples.
            **kwargs: Subclass-specific options forwarded to setup/loading
                hooks.
        """
        super().__init__()
        self.samples: List[SampleDict] = []
        self.root = Path(root)
        self.split = split
        self.seq_len = seq_len
        self.data_dir = self.root / split

        self._setup(**kwargs)
        self._discover_ids()
        self.samples = self._load_samples(**kwargs)

    def _setup(self, **kwargs):
        """Configure subclass-specific paths or state before sample loading.

        Subclasses may override this hook when the default ``root / split`` data
        directory convention is not enough.
        """
        pass

    @abstractmethod
    def _load_samples(self, **kwargs) -> List[SampleDict]:
        """Load sample dictionaries for the active split."""
        raise NotImplementedError("Subclasses must implement _load_samples method.")

    def __len__(self):
        """Return the number of loaded samples."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> SampleDict:
        """Return one sample with waveform arrays converted to tensors.

        Labels are converted to ``torch.long`` for ``CrossEntropyLoss``. Waveform
        arrays are converted to ``torch.float32``. Metadata fields are returned
        unchanged.
        """
        sample = self.samples[idx]

        def _to_tensor(data: Union[np.ndarray, torch.Tensor], *, is_label: bool = False) -> torch.Tensor:
            """Convert numpy arrays to tensors while preserving existing tensors."""
            if isinstance(data, torch.Tensor):
                return data
            if is_label:
                return torch.tensor(data, dtype=torch.long)
            return torch.tensor(data, dtype=torch.float32)

        return {
            'x_hw': _to_tensor(sample['x_hw']),
            'x_bw': _to_tensor(sample['x_bw']),
            'label': _to_tensor(sample['label'], is_label=True),
            'subject_id': sample['subject_id'],
            'start_idx': sample['start_idx'],
        }

    def get_class_weights(self) -> torch.Tensor:
        """Compute inverse-frequency class weights from loaded labels.

        Returns:
            Float tensor with one weight per observed class. The weights are
            normalized so their sum equals the number of observed classes.
        """
        labels_np: List[np.ndarray] = []
        for sample in self.samples:
            y = sample["label"]
            if isinstance(y, torch.Tensor):
                y = y.detach().cpu().numpy()
            labels_np.append(np.asarray(y).reshape(-1))

        all_labels = np.concatenate(labels_np, axis=0)
        classes, counts = np.unique(all_labels, return_counts=True)
        weights = 1.0 / counts.astype(np.float64)
        weights = weights / weights.sum() * len(classes)
        return torch.tensor(weights, dtype=torch.float32)

    def get_subject_ids(self) -> List[str]:
        """Return sorted unique subject or recording identifiers."""
        return sorted({sample.get("subject_id", "") for sample in self.samples})

    def get_stats(self) -> Dict[str, Any]:
        """Summarize the loaded split.

        Returns:
            Dictionary containing sample count, subject count, sequence length,
            total label count and per-class label distribution.
        """
        n_samples = len(self.samples)
        n_subjects = len(self.get_subject_ids())

        labels_np: List[np.ndarray] = []
        for sample in self.samples:
            y = sample["label"]
            if isinstance(y, torch.Tensor):
                y = y.detach().cpu().numpy()
            labels_np.append(np.asarray(y).reshape(-1))
        all_labels = np.concatenate(labels_np, axis=0) if labels_np else np.array([], dtype=np.int64)

        if all_labels.size:
            classes, counts = np.unique(all_labels, return_counts=True)
            class_dist = {int(c): int(n) for c, n in zip(classes, counts)}
        else:
            class_dist = {}

        return {
            "split": getattr(self, "split", "unknown"),
            "n_samples": n_samples,
            "n_subjects": n_subjects,
            "sequence_length": getattr(self, "seq_len", -1),
            "total_epochs": int(all_labels.size),
            "class_distribution": class_dist,
        }

    @abstractmethod
    def _discover_ids(self) -> List[str]:
        """Return record IDs belonging to the active split."""
        raise NotImplementedError("Subclasses must implement _discover_ids method.")

    @staticmethod
    def parse_xml(xml_path):
        """Parse NSRR-style XML sleep-stage annotations.

        Args:
            xml_path: Path to an XML annotation file containing ``ScoredEvent``
                entries.

        Returns:
            List of dictionaries with ``start``, ``duration`` and integer
            ``label`` fields. Stage 1 and 2 are merged into class 1, stage 3 and
            4 into class 2, and unscored/movement epochs are mapped to Wake.
        """
        tree = ET.parse(xml_path)
        root = tree.getroot()

        label_map = {
            "Wake": 0,
            "Stage 1 sleep": 1,
            "Stage 2 sleep": 1,
            "Stage 3 sleep": 2,
            "Stage 4 sleep": 2,
            "REM sleep": 3,
            "Unscored": 0,
            "Movement": 0,
        }

        sleep_epochs = []
        for event in root.iter('ScoredEvent'):
            event_type = event.find('EventType').text
            if 'Stages' not in str(event_type):
                continue

            concept = event.find('EventConcept').text
            start = float(event.find('Start').text)
            duration = float(event.find('Duration').text)
            for label in label_map.keys():
                if label in concept:
                    concept = label
                    break
            if concept is None:
                continue

            stage_label = label_map[concept]
            n_epochs = int(duration // 30)
            for i in range(n_epochs):
                sleep_epochs.append({
                    "start": start + i * 30,
                    "duration": 30,
                    "label": stage_label,
                })

        return sleep_epochs

    @staticmethod
    def get_last_epoch_fps(json_path):
        """Read recording duration metadata from a KVSS annotation JSON file.

        Args:
            json_path: Path to a KVSS annotation JSON file.

        Returns:
            Tuple ``(total_epoch, fps, record_id)`` where ``total_epoch`` is the
            number of complete 30-second epochs covered by the video metadata.
        """
        with open(json_path, 'r') as f:
            ann = json.load(f)
            fps = ann["Video_Info"][0]["Frame_Rate"]
            record_id = ann["Case_Info"]["Case_Number"]
            start_time = datetime.datetime.strptime(ann['Video_Info'][0]['Start'], "%Y/%m/%d %H:%M:%S.%f")
            end_time = datetime.datetime.strptime(ann['Video_Info'][0]['End'], "%Y/%m/%d %H:%M:%S.%f")
            total_epoch = (end_time - start_time).seconds // 30
            return total_epoch, fps, record_id

    @staticmethod
    def parse_json(json_path):
        """Parse KVSS JSON sleep-stage annotations.

        Args:
            json_path: Path to a KVSS annotation JSON file.

        Returns:
            Tuple ``(fps, sleep_epochs)``. ``sleep_epochs`` is a list of
            dictionaries with ``start``, ``duration`` and integer ``label``
            fields. N1 and N2 are merged into class 1.
        """
        label_map = {
            "Wake": 0,
            "N1": 1,
            "N2": 1,
            "N3": 2,
            "REM": 3,
        }
        sleep_epochs = []
        with open(json_path, 'r') as f:
            ann = json.load(f)
            fps = ann["Video_Info"][0]["Frame_Rate"]

            for row in ann['Event']:
                if row['Event_Label'] not in label_map:
                    continue
                sleep_epochs.append({
                    "start": int((row['Start_Epoch']) - 1 * 30),
                    "duration": int(row["Duration(second)"]),
                    "label": label_map[row['Event_Label']],
                })
            return fps, sleep_epochs

    @staticmethod
    def parse_csv(csv_path):
        """Parse CSV sleep-stage annotations.

        Args:
            csv_path: Path to a CSV file with ``Event_Label`` and
                ``Start_Epoch`` columns.

        Returns:
            List of dictionaries with ``start``, ``duration`` and integer
            ``label`` fields. N1 and N2 are merged into class 1.
        """
        label_map = {
            "Wake": 0,
            "N1": 1,
            "N2": 1,
            "N3": 2,
            "REM": 3,
        }
        sleep_epochs = []
        with open(csv_path, 'r') as f:
            ann = [row for row in csv.DictReader(f)]

            for row in ann:
                if row['Event_Label'] not in label_map:
                    continue
                sleep_epochs.append({
                    "start": (float(row['Start_Epoch']) - 1) * 30,
                    "duration": 30,
                    "label": label_map[row['Event_Label']],
                })
            return sleep_epochs

    @staticmethod
    def collate_fn(
        batch: Sequence[Dict[str, Any]],
        pad_value: float = 0.0,
        stack_labels: bool = True,
        return_dict: bool = True,
    ):
        """Pad variable-length waveform samples for batched inference.

        Args:
            batch: Sequence of sample dictionaries containing ``x_hw``, ``x_bw``
                and ``label`` arrays or tensors.
            pad_value: Value used to pad waveform arrays on the time axis.
            stack_labels: If ``True``, labels are padded with ``-100`` and
                stacked. ``-100`` matches PyTorch's common ``ignore_index`` for
                cross-entropy loss.
            return_dict: If ``True``, return a dictionary. Otherwise return a
                tuple for legacy call sites.

        Returns:
            Either a padded batch dictionary or ``(x_hw, x_bw, labels, lengths,
            subject_ids, start_idxs)``.
        """
        x_hw_list = [b["x_hw"] for b in batch]
        x_bw_list = [b["x_bw"] for b in batch]
        label_list = [b["label"] for b in batch]
        subject_ids = [b.get("subject_id", "") for b in batch]
        start_idxs = [int(b.get("start_idx", 0)) for b in batch]

        max_len_hw = max(x.shape[0] for x in x_hw_list)
        max_len_bw = max(x.shape[0] for x in x_bw_list)
        max_len_label = max(x.shape[0] for x in label_list)

        padded_x_hw: List[torch.Tensor] = []
        padded_x_bw: List[torch.Tensor] = []
        padded_labels: List[torch.Tensor] = []

        for hw, bw, lb in zip(x_hw_list, x_bw_list, label_list):
            if not isinstance(hw, torch.Tensor):
                hw = torch.as_tensor(hw, dtype=torch.float32)
            if not isinstance(bw, torch.Tensor):
                bw = torch.as_tensor(bw, dtype=torch.float32)
            if not isinstance(lb, torch.Tensor):
                lb = torch.as_tensor(lb, dtype=torch.long)

            pad_hw = max_len_hw - hw.shape[0]
            pad_bw = max_len_bw - bw.shape[0]
            pad_lb = max_len_label - lb.shape[0]

            if pad_hw > 0:
                hw_pad = torch.full((pad_hw, hw.shape[1]), pad_value, dtype=hw.dtype, device=hw.device)
                hw = torch.cat([hw, hw_pad], dim=0)
            if pad_bw > 0:
                bw_pad = torch.full((pad_bw, bw.shape[1]), pad_value, dtype=bw.dtype, device=bw.device)
                bw = torch.cat([bw, bw_pad], dim=0)
            if pad_lb > 0 and stack_labels:
                lb_pad = torch.full((pad_lb,), -100, dtype=lb.dtype, device=lb.device)
                lb = torch.cat([lb, lb_pad], dim=0)

            padded_x_hw.append(hw)
            padded_x_bw.append(bw)
            padded_labels.append(lb)

        x_hw_batch = torch.stack(padded_x_hw, dim=0)
        x_bw_batch = torch.stack(padded_x_bw, dim=0)
        labels_batch = torch.stack(padded_labels, dim=0) if stack_labels else padded_labels
        lengths = torch.tensor([x.shape[0] for x in x_hw_list])

        if return_dict:
            return {
                "x_hw": x_hw_batch,
                "x_bw": x_bw_batch,
                "label": labels_batch,
                "lengths": lengths,
                "subject_ids": subject_ids,
                "start_idxs": start_idxs,
            }
        return x_hw_batch, x_bw_batch, labels_batch, lengths, subject_ids, start_idxs

    @classmethod
    def create_subset(
        cls,
        full_dataset: "BaseDataset",
        max_samples: Optional[int] = None,
        max_subjects: Optional[int] = None,
        seed: int = 42,
    ) -> "BaseDataset":
        """Create a shallow subset without re-reading files from disk.

        Args:
            full_dataset: Dataset whose already-loaded samples should be
                filtered.
            max_samples: Optional maximum number of samples to keep.
            max_subjects: Optional maximum number of unique subject IDs to keep.
            seed: Random seed used when selecting subjects or samples.

        Returns:
            Dataset-like object of the same class with copied metadata and a
            filtered ``samples`` list. The subclass initializer is intentionally
            bypassed to avoid expensive file IO.
        """
        rng = np.random.default_rng(seed)
        subset = cls.__new__(cls)

        subset.seq_len = getattr(full_dataset, "seq_len", -1)
        subset.split = getattr(full_dataset, "split", "subset")
        samples: List[SampleDict] = list(full_dataset.samples)

        if max_subjects is not None:
            subject_ids = list({s.get("subject_id", "") for s in samples})
            rng.shuffle(subject_ids)
            keep = set(subject_ids[: max(0, max_subjects)])
            samples = [s for s in samples if s.get("subject_id", "") in keep]

        if max_samples is not None and len(samples) > max_samples:
            idx = rng.choice(len(samples), size=max_samples, replace=False)
            samples = [samples[i] for i in idx]

        subset.samples = samples
        return subset


class BaseDataModule:
    """Minimal datamodule wrapper around one dataset instance.

    The project mostly instantiates datasets directly, but some legacy paths use
    this class to keep dataloader construction options in one place.
    """

    def __init__(
        self,
        cfg: DictConfig,
        dataset: BaseDataset,
    ) -> None:
        """Store dataloader options from a Hydra config.

        Args:
            cfg: Config node containing dataloader fields such as ``batch_size``
                and ``num_workers``.
            dataset: Dataset instance to wrap.
        """
        self.cfg = cfg.data
        self.dataset = dataset
        self._collate_fn = self.dataset.collate_fn
        self.shuffle = cfg.get("shuffle", True)
        self.batch_size = cfg.get("batch_size", 1)
        self.pin_memory = cfg.get("pin_memory", True)
        self.num_workers = cfg.get("num_workers", 8)

    def train_dataloader(self) -> DataLoader:
        """Build a training dataloader using configured shuffle behavior."""
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self._collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        """Build a validation dataloader using configured options."""
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self._collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        """Build a test dataloader using configured options."""
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self._collate_fn,
        )
