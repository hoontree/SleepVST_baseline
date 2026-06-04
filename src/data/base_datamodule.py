from __future__ import annotations
from torch.utils.data import DataLoader, Dataset
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Union, TypedDict
from omegaconf import DictConfig
from pathlib import Path

import json
import datetime
import numpy as np
import torch
from pathlib import Path
import csv
import xml.etree.ElementTree as ET

class SampleDict(TypedDict, total=False):
    """
    공통 샘플 포맷 (common sample format)
    - x_hw: (T, H) numpy.ndarray 또는 torch.Tensor
    - x_bw: (T, B) numpy.ndarray 또는 torch.Tensor
    - label: (T,) 또는 (T, C) numpy.ndarray/torch.Tensor
    - subject_id: str (피험자 ID)
    - start_idx: int (시퀀스 시작 위치)
    """
    x_hw: Union[np.ndarray, torch.Tensor]
    x_bw: Union[np.ndarray, torch.Tensor]
    label: Union[np.ndarray, torch.Tensor]
    subject_id: str
    start_idx: int

class BaseDataset(Dataset, ABC):
    seq_len: int
    split: str
    data_dir: str

    def __init__(self, root, split, seq_len, **kwargs):
        super().__init__()
        self.samples: List[SampleDict] = []
        self.root = Path(root)
        self.split = split
        self.seq_len = seq_len
        self.data_dir = self.root / split
        
        # 서브클래스에서 오버라이드할 수 있도록 분리
        self._setup(**kwargs)
        ids = self._discover_ids()
        self.samples = self._load_samples(**kwargs)
    
    def _setup(self, **kwargs):
        """서브클래스에서 추가 설정을 위해 오버라이드"""
        pass
        
    @abstractmethod
    def _load_samples(self, **kwargs) -> List[SampleDict]:
        """서브클래스에서 구현해야 하는 샘플 로딩 메소드"""
        raise NotImplementedError("Subclasses must implement _load_samples method.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> SampleDict:
        sample = self.samples[idx]
        
        def _to_tensor(data: Union[np.ndarray, torch.Tensor], *, is_label: bool = False) -> torch.Tensor:
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
            'start_idx': sample['start_idx']
        }

    def get_class_weights(self) -> torch.Tensor:
        """
        클래스 불균형 보정용 weight 계산.
        - label이 (T,) 정수 클래스라고 가정(CrossEntropyLoss).
        """
        labels_np: List[np.ndarray] = []
        for s in self.samples:
            y = s["label"]
            if isinstance(y, torch.Tensor):
                y = y.detach().cpu().numpy()
            labels_np.append(np.asarray(y).reshape(-1))
        all_labels = np.concatenate(labels_np, axis=0)

        classes, counts = np.unique(all_labels, return_counts=True)
        weights = 1.0 / counts.astype(np.float64)
        weights = weights / weights.sum() * len(classes)  # normalize
        return torch.tensor(weights, dtype=torch.float32)
    
    def get_subject_ids(self) -> List[str]:
        return sorted(list({s.get("subject_id", "") for s in self.samples}))

    def get_stats(self) -> Dict[str, Any]:
        """데이터 개요 통계치."""
        n_samples = len(self.samples)
        n_subjects = len(self.get_subject_ids())

        labels_np: List[np.ndarray] = []
        for s in self.samples:
            y = s["label"]
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
        """
        서브클래스에서 구현해야 하는 메소드.
        - 데이터셋 split 정의에서 레코드 id 목록을 반환해야 함.
        """
        raise NotImplementedError("Subclasses must implement _discover_ids method.")
        
    def parse_xml(xml_path):
        """
        Args:
            xml_path (str)
        Returns:
            list: 수면 단계 정보가 포함된 딕셔너리 리스트
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
            if concept == None:
                continue
            stage_label = label_map[concept]

            n_epochs = int(duration // 30)
            for i in range(n_epochs):
                sleep_epochs.append({
                    "start": start + i * 30,
                    "duration": 30,
                    "label": stage_label
                })

        return sleep_epochs
    
    def get_last_epoch_fps(json_path):
        """
        Args:
            json_path (str)
        Returns:
            int: 마지막 에포크 번호
        """

        with open(json_path, 'r') as f:
            ann = json.load(f)
            fps = ann["Video_Info"][0]["Frame_Rate"]
            record_id = ann["Case_Info"]["Case_Number"]
            start_time = datetime.datetime.strptime(ann['Video_Info'][0]['Start'], "%Y/%m/%d %H:%M:%S.%f")
            end_time = datetime.datetime.strptime(ann['Video_Info'][0]['End'], "%Y/%m/%d %H:%M:%S.%f")
            total_epoch = (end_time - start_time).seconds // 30
            return total_epoch, fps, record_id

    def parse_json(json_path):
        """
        Args:
            json_path (str)
        Returns:
            list: 수면 단계 정보가 포함된 딕셔너리 리스트
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
                    "start": int((row['Start_Epoch'])-1 * 30),
                    "duration": int(row["Duration(second)"]),
                    "label": label_map[row['Event_Label']]
                })
            return fps, sleep_epochs
    
    def parse_csv(csv_path):
        """
        Args:
            csv_path (str)
        Returns:
            list: 수면 단계 정보가 포함된 딕셔너리 리스트
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
                    "start": (float(row['Start_Epoch'])-1) * 30,
                    "duration": 30,
                    "label": label_map[row['Event_Label']]
                })
            return sleep_epochs
    
    # -------- DataLoader 전용 collate_fn --------
    @staticmethod
    def collate_fn(
        batch: Sequence[Dict[str, Any]],
        pad_value: float = 0.0,
        stack_labels: bool = True,
        return_dict: bool = True,
    ):
        """
        길이가 다른 시퀀스를 뒤쪽 패딩(post-pad).
        - x_hw: (T, H), x_bw: (T, B), label: (T,)
        - label 패딩은 CrossEntropyLoss의 ignore_index=-100 관례를 따름.
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
                lb = torch.as_tensor(lb, dtype=torch.long)  # 정수 클래스

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
                lb_pad = torch.full((pad_lb,), -100, dtype=lb.dtype, device=lb.device)  # ignore_index
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
        else:
            return x_hw_batch, x_bw_batch, labels_batch, lengths, subject_ids, start_idxs

    # -------- 공통 서브셋 유틸리티 --------
    @classmethod
    def create_subset(
        cls,
        full_dataset: "BaseDataset",
        max_samples: Optional[int] = None,
        max_subjects: Optional[int] = None,
        seed: int = 42,
    ) -> "BaseDataset":
        """
        기존 데이터셋에서 샘플/피험자 수를 제한한 서브셋 객체 생성.
        반환 객체는 같은 클래스(cls)의 얕은 복제이며, 파일 로드는 수행하지 않음.
        """
        rng = np.random.default_rng(seed)

        subset = cls.__new__(cls)  # __init__ 호출 없이 생성

        # 메타 복사
        subset.seq_len = getattr(full_dataset, "seq_len", -1)
        subset.split = getattr(full_dataset, "split", "subset")

        # 전체 샘플 복사
        samples: List[SampleDict] = list(full_dataset.samples)

        # 피험자 제한
        if max_subjects is not None:
            subject_ids = list({s.get("subject_id", "") for s in samples})
            rng.shuffle(subject_ids)
            keep = set(subject_ids[: max(0, max_subjects)])
            samples = [s for s in samples if s.get("subject_id", "") in keep]

        # 샘플 수 제한
        if max_samples is not None and len(samples) > max_samples:
            idx = rng.choice(len(samples), size=max_samples, replace=False)
            samples = [samples[i] for i in idx]

        subset.samples = samples
        return subset


class BaseDataModule:
    """
    - 공통 DataLoader 옵션 보관 및 생성
    - collate_fn은 기본적으로 BaseDataset.collate_fn 사용(교체 가능)
    """

    def __init__(
        self,
        cfg: DictConfig,
        dataset: BaseDataset,
    ) -> None:
        self.cfg = cfg.data
        self.dataset = dataset
        self._collate_fn = self.dataset.collate_fn
        self.shuffle = cfg.get("shuffle", True)
        self.batch_size = cfg.get("batch_size", 1)
        self.pin_memory = cfg.get("pin_memory", True)
        self.num_workers = cfg.get("num_workers", 8)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self._collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self._collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self._collate_fn,
        )