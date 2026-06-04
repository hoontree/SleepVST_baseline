from torch.utils.data import DataLoader, Dataset
from src.data.base_datamodule import BaseDataset, SampleDict, BaseDataModule
from pathlib import Path
from tqdm import tqdm
from typing import List

import scipy.signal

import numpy as np

class KVSS(BaseDataset):
    def __init__(self, cfg):
        print(f"KVSS Dataset Configuration:")
        print(f"  Split: {cfg.data.split}")
        print(f"  Root: {cfg.data.root}")
        print(f"  Motion dir: {cfg.data.motion_dir}")
        print(f"  Signal dir: {cfg.data.signal_dir}")
        print(f"  Label dir: {cfg.data.label_dir}")
        
        self.seq_len = cfg.data.seq_len
        self.split = cfg.data.split
        self.data_dir = Path(cfg.data.root)
        self.motion_dir = cfg.data.motion_dir
        self.signal_dir = cfg.data.signal_dir
        self.label_dir = Path(cfg.data.label_dir)
        self.video_dir = Path(cfg.data.video_dir)
        self.respiratory_signal_dir = Path(cfg.data.respiratory_signal_dir)
        
        # Motion feature key 정보를 저장할 속성 추가
        self.motion_feature_keys = None
        
        self.exceptions = set(cfg.data.get('exceptions', []))
        self.cfg = cfg
        super().__init__(cfg.data.root, cfg.data.split, cfg.data.seq_len, cfg=cfg)
    def _setup(self, cfg, **kwargs):
        if cfg.mode.name == 'transfer':
            self.signal_dir = Path(self.signal_dir)
            self.motion_dir = Path(self.motion_dir)
            self.motion_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.signal_dir = Path(self.signal_dir)
        
        # 디렉토리 생성
        self.signal_dir.mkdir(parents=True, exist_ok=True)
    
    def _load_samples(self, cfg, **kwargs) -> List[SampleDict]:
        if cfg.mode.name == 'transfer':
            if 'BW' in cfg.model.name:
                print("Loading motion features for BW model...")
                return self.load_motion_samples_bw()
            else:
                print("Loading motion features for general model...")
                return self.load_motion_samples()
        else:
            return self.load_regular_samples(cfg)

    def _discover_ids(self):
        """
        레코드 파일 목록을 반환합니다.
        KVSS 데이터셋의 경우, 'A-train_set.txt', 'A-valid_set.txt', 'A-test_set.txt' 파일에서 ID를 읽어옵니다.
        """
        if self.split == 'train':
            train_list_file = self.root / 'A-train_set.txt'
            with open(train_list_file, 'r') as f:
                original_list = set([line.strip().replace('.h5', '') for line in f.readlines()])
            print(f"exceptions: {self.exceptions}")
            excluded = original_list & self.exceptions
            self.train_list = original_list - self.exceptions
            
            if excluded:
                print(f"\n[{self.split.upper()}] Excluded {len(excluded)} samples due to exceptions:")
                for sample_id in sorted(excluded):
                    print(f"  - {sample_id}")
            print(f"[{self.split.upper()}] Total samples: {len(original_list)} -> {len(self.train_list)} (excluded: {len(excluded)})")
            
            return self.train_list
            
        elif self.split == 'valid':
            valid_list_file = self.root / 'A-valid_set.txt'
            with open(valid_list_file, 'r') as f:
                original_list = set([line.strip().replace('.h5', '') for line in f.readlines()])
            print(f"exceptions: {self.exceptions}")
            excluded = original_list & self.exceptions
            self.valid_list = original_list - self.exceptions
            
            if excluded:
                print(f"\n[{self.split.upper()}] Excluded {len(excluded)} samples due to exceptions:")
                for sample_id in sorted(excluded):
                    print(f"  - {sample_id}")
            print(f"[{self.split.upper()}] Total samples: {len(original_list)} -> {len(self.valid_list)} (excluded: {len(excluded)})")
            
            return self.valid_list
            
        elif self.split == 'test':
            test_list_file = self.root / 'A-test_set.txt'
            with open(test_list_file, 'r') as f:
                original_list = set([line.strip().replace('.h5', '') for line in f.readlines()])
            print(f"exceptions: {self.exceptions}")
            excluded = original_list & self.exceptions
            self.test_list = original_list - self.exceptions
            
            if excluded:
                print(f"\n[{self.split.upper()}] Excluded {len(excluded)} samples due to exceptions:")
                for sample_id in sorted(excluded):
                    print(f"  - {sample_id}")
            print(f"[{self.split.upper()}] Total samples: {len(original_list)} -> {len(self.test_list)} (excluded: {len(excluded)})")
            
            return self.test_list
            
        else:
            raise ValueError(f"Unknown split: {self.split}")

    def load_regular_samples(self, cfg) -> List[SampleDict]:
        samples = []
        if cfg.data.split == 'test':
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
                sample: SampleDict = {
                    "x_hw": x_hw[:T],
                    "x_bw": x_bw[:T],
                    "label": labels[:T],
                    "subject_id": record_id,
                    "start_idx": 0
                }
                samples.append(sample)
                
            return samples
        else:
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
                    sample: SampleDict = {
                        "x_hw": x_hw[i:i + self.seq_len],
                        "x_bw": x_bw[i:i + self.seq_len],
                        "label": labels[i:i + self.seq_len],
                        "subject_id": record_id,
                        "start_idx": i
                    }
                    samples.append(sample)
                    
            return samples
    
    def _create_temporal_features(self, feature_list, motion_keys=None):
        """
        Motion feature keys 정보를 포함하여 temporal features 생성
        """
        N, patch_size, num_feature = feature_list.shape
        
        feature_list = np.mean(feature_list, axis=1)
        
        # 패딩을 미리 계산
        padded_features = np.pad(feature_list, ((3, 3), (0, 0)), mode='edge')

        # 슬라이싱으로 past, current, future 추출
        past_features = padded_features[0:N]
        current_features = padded_features[3:N+3]
        future_features = padded_features[6:N+6]
        
        # 한번에 concatenate
        expanded = np.concatenate([past_features, current_features, future_features], axis=1)
        
        # Motion feature keys 정보 업데이트 (첫 번째 호출에서만)
        if motion_keys is not None and self.motion_feature_keys is None:
            # past, current, future에 대해 각각 key 생성
            temporal_keys = []
            for prefix in ['past', 'current', 'future']:
                for key in motion_keys:
                    temporal_keys.append(f"{prefix}_{key}")
            self.motion_feature_keys = temporal_keys
        
        return expanded

    def load_motion_samples_bw(self) -> List[SampleDict]:
        samples = []
        discovered_ids = self._discover_ids()
        print(f"Discovered {len(discovered_ids)} IDs for split '{self.split}'")
        
        for record_id in tqdm(discovered_ids):
            motion_file = self.motion_dir / f"{record_id}_motion_features.npy"
            label_csv = self.label_dir / f"{record_id}_label.csv"
            label_json = self.video_dir / record_id / f"{record_id}_annotation.json"


            fps, epochs = BaseDataset.parse_json(label_json)
            # x_bw = np.load(signal_file).astype(np.float32)
            x_bw_list = []
            if self.cfg.data.data_source == 'video':

                root = Path(self.respiratory_signal_dir / record_id)
                signal_files = list(root.glob('epoch_*/*.npy'))

                def extract_num(path: Path):
                    epoch_dir = path.parent.name
                    epoch_num = int(epoch_dir.split('_')[1])
                    return epoch_num

                signal_files = sorted(signal_files, key=extract_num)
                dim = 74  # 기대하는 차원
                
                if not all([signal_files, label_json.exists(), motion_file.exists()]):
                    continue
                if signal_files:
                    last = signal_files[-1]
                    arr = np.load(last).astype(np.float32)
                    if arr.shape[0] != 74:
                        signal_files = signal_files[:-1]  # 마지막 파일 제외
                for signal_file in signal_files:
                    x_bw_part = np.load(signal_file).astype(np.float32)
                    if x_bw_part.shape[0] != 74:
                        x_bw_part = scipy.signal.resample_poly(x_bw_part, 74, x_bw_part.shape[0])
                    x_bw_list.append(x_bw_part)
                x_bw = np.vstack(x_bw_list)
            else:  # raw signal 사용
                signal_file = self.signal_dir / f"{record_id}_bw.npy"
                if not all([signal_file.exists(), label_json.exists(), motion_file.exists()]):
                    print(f"Missing files for {record_id}, skipping...")
                    continue
                raw_signal_file = self.signal_dir / f"{record_id}_bw.npy"
                x_bw = np.load(raw_signal_file).astype(np.float32)
            motion = np.load(motion_file, allow_pickle=True).item()
    
            # motion features 처리 - key 정보 보존
            motion_features = []
            motion_keys = sorted(motion.keys())  # key 순서 고정
            for key in motion_keys:
                patches = self.patchify(motion[key], int(fps * 30)) # (N, patch_size)
                motion_features.append(patches)
                         
            expanded_motion_features = self._create_temporal_features(
                np.array(motion_features).transpose(1, 2, 0), 
                motion_keys=motion_keys
            )  # (N, n_features)

            labels = np.array([e['label'] for e in epochs], dtype=np.int64)

            T = min(len(x_bw), len(expanded_motion_features), len(labels))
            
            sample = {
                "x_bw": x_bw[:T],
                "motion": expanded_motion_features[:T],
                "label": labels[:T],
                "subject_id": record_id,
                "start_idx": 0
            }
            samples.append(sample)
    
        print(f"Total samples loaded: {len(samples)}")
        return samples
    
    def load_motion_samples(self) -> List[SampleDict]:
        samples = []
        discovered_ids = self._discover_ids()
        print(f"Discovered {len(discovered_ids)} IDs for split '{self.split}'")
        
        for record_id in tqdm(discovered_ids):
            hw_signal_file = self.signal_dir / f"{record_id}_hw.npy"
            motion_file = self.motion_dir / f"{record_id}_motion_features.npy"
            label_csv = self.label_dir / f"{record_id}_label.csv"
            label_json = self.video_dir / record_id / f"{record_id}_annotation.json"

            fps, epochs = BaseDataset.parse_json(label_json)
            
            x_hw = np.load(hw_signal_file).astype(np.float32)
            x_bw_list = []
            if self.cfg.data.data_source == 'video':

                root = Path(self.respiratory_signal_dir / record_id)
                signal_files = list(root.glob('epoch_*/*.npy'))

                def extract_num(path: Path):
                    epoch_dir = path.parent.name
                    epoch_num = int(epoch_dir.split('_')[1])
                    return epoch_num

                signal_files = sorted(signal_files, key=extract_num)
                dim = 74  # 기대하는 차원
                
                if not all([signal_files, label_json.exists(), motion_file.exists()]):
                    continue
                if signal_files:
                    last = signal_files[-1]
                    arr = np.load(last).astype(np.float32)
                    if arr.shape[0] != 74:
                        signal_files = signal_files[:-1]  # 마지막 파일 제외
                for signal_file in signal_files:
                    x_bw_part = np.load(signal_file).astype(np.float32)
                    if x_bw_part.shape[0] != 74:
                        x_bw_part = scipy.signal.resample_poly(x_bw_part, 74, x_bw_part.shape[0])
                    x_bw_list.append(x_bw_part)
                x_bw = np.vstack(x_bw_list)
            else:  # raw signal 사용
                signal_file = self.signal_dir / f"{record_id}_bw.npy"
                if not all([signal_file.exists(), label_json.exists(), motion_file.exists()]):
                    print(f"Missing files for {record_id}, skipping...")
                    continue
                raw_signal_file = self.signal_dir / f"{record_id}_bw.npy"
                x_bw = np.load(raw_signal_file).astype(np.float32)
            motion = np.load(motion_file, allow_pickle=True).item()
            
            # motion features 처리 - key 정보 보존
            motion_features = []
            motion_keys = sorted(motion.keys())  # key 순서 고정
            for key in motion_keys:
                patches = self.patchify(motion[key], patch_size=int(fps * 30)) # (N, patch_size)
                motion_features.append(patches)
                
            motion_features = np.array(motion_features)  # (n_features, N, patch_size)
            expanded_motion_features = self._create_temporal_features(
                motion_features.transpose(1, 2, 0),
                motion_keys=motion_keys
            )  # (N, n_features)

            labels = np.array([e['label'] for e in epochs], dtype=np.int64)
            
            # 최소 길이로 맞추기
            T = min(len(x_bw), len(motion_features[0]), len(labels))
            
            sample = {
                "x_bw": x_bw[:T],
                "x_hw": x_hw[:T],
                "motion": expanded_motion_features[:T,],
                "label": labels[:T],
                "subject_id": record_id,
                "start_idx": 0
            }
            samples.append(sample)
    
        print(f"Total samples loaded: {len(samples)}")
        return samples

    def get_motion_feature_keys(self):
        """Motion feature keys 반환"""
        return self.motion_feature_keys

    def __getitem__(self, idx):
        return self.samples[idx]

    def normalize(self, patch):
        return (patch - np.mean(patch)) / (np.std(patch) + 1e-6)
    
    def patchify(self, signal, patch_size):
        """신호에서 패치를 추출합니다."""
        patches = []
        for start in range(0, len(signal) - patch_size + 1, patch_size):
            patch = signal[start:start + patch_size]
            # patch = self.normalize(patch)
            patches.append(patch)
        return np.stack(patches)  # shape: (N, patch_size)
    
    def patch_mean(self, signal, patch_size):
        """신호에서 패치의 평균을 계산합니다."""
        patches = self.patchify(signal, patch_size)
        return np.mean(patches, axis=1) # (N,)

class KVSSDataModule(BaseDataModule):
    def __init__(self, cfg):
        super().__init__(cfg, KVSS)
        self.dataset = KVSS(cfg)

    def get_dataloader(self):
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