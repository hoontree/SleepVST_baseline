from torch.utils.data import DataLoader, Dataset
from src.data.base_datamodule import BaseDataset, SampleDict, BaseDataModule
from pathlib import Path
from tqdm import tqdm
from typing import List, Union
from omegaconf import DictConfig

import numpy as np

class SHHS(BaseDataset):
    def __init__(self, cfg: Union[DictConfig, dict] = None, split: str = None, **kwargs):
        """
        Args:
            cfg: Configuration object or dict with 'root', 'seq_len', etc.
            split: Split name ('train', 'val', 'test'). Overrides cfg.split if provided.
            **kwargs: Additional arguments (data_dir, seq_len for backward compatibility)
        """
        super().__init__()
        
        # Backward compatibility: support old (data_dir, split, seq_len) signature
        if cfg is None:
            # Old style: SHHS(data_dir, split, seq_len=30)
            data_dir = kwargs.get('data_dir', split)  # split might be data_dir in old call
            self.split = kwargs.get('split', 'train')
            self.seq_len = kwargs.get('seq_len', 30)
            self.data_dir = Path(data_dir) / self.split
        else:
            # New style: SHHS(cfg, split='train')
            if hasattr(cfg, 'root'):
                root = cfg.root
            elif isinstance(cfg, dict) and 'root' in cfg:
                root = cfg['root']
            else:
                root = cfg
                
            self.split = split if split is not None else getattr(cfg, 'split', 'train')
            self.seq_len = getattr(cfg, 'seq_len', 240)
            self.data_dir = Path(root) / self.split
        
        self.samples = self.load_samples()
        self.exceptions = set()

    def _discover_ids(self):
        """
        레코드 파일 목록을 반환합니다.
        KVSS 데이터셋의 경우, 'A-train_set.txt', 'A-valid_set.txt', 'A-test_set.txt' 파일에서 ID를 읽어옵니다.
        """
        if self.split == 'train':
            train_list_file = self.data_dir / 'A-train_set.txt'
            with open(train_list_file, 'r') as f:
                self.train_list = set([line.strip().replace('.h5', '') for line in f.readlines()]) - self.exceptions
            return self.train_list
        elif self.split == 'valid':
            valid_list_file = self.data_dir / 'A-valid_set.txt'
            with open(valid_list_file, 'r') as f:
                self.valid_list = set([line.strip().replace('.h5', '') for line in f.readlines()]) - self.exceptions
            return self.valid_list
        elif self.split == 'test':
            test_list_file = self.data_dir / 'A-test_set.txt'
            with open(test_list_file, 'r') as f:
                self.test_list = set([line.strip().replace('.h5', '') for line in f.readlines()]) - self.exceptions
            return self.test_list
        else:
            raise ValueError(f"Unknown split: {self.split}")

    def load_samples(self, step=10, is_test: bool = False) -> List[SampleDict]:
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
                sample: SampleDict = {
                    "x_hw": x_hw[:T],
                    "x_bw": x_bw[:T],
                    "label": labels[:T],
                    "subject_id": record_id,
                    "start_idx": 0
                }
                samples.append(sample)
                
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
                sample: SampleDict = {
                    "x_hw": x_hw[i:i + self.seq_len],
                    "x_bw": x_bw[i:i + self.seq_len],
                    "label": labels[i:i + self.seq_len],
                    "subject_id": record_id,
                    "start_idx": i
                }
                samples.append(sample)
                
        return samples

class SHHSDataModule(BaseDataModule):
    def __init__(self, data_dir, batch_size=32, seq_len=30):
        super().__init__(data_dir, batch_size)
        self.seq_len = seq_len

    def setup(self, stage=None):
        self.train_dataset = SHHS(self.data_dir, 'train', self.seq_len)
        self.valid_dataset = SHHS(self.data_dir, 'valid', self.seq_len)
        self.test_dataset = SHHS(self.data_dir, 'test', self.seq_len)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)