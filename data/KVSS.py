import numpy as np
import os.path as path
import torch
import torch.utils.data as data
import random
import re
from utils.util import parse_csv
from glob import glob
from utils.customlogger import Logger
from config import parse_args
from tqdm import tqdm

# 로깅 설정
logger = Logger(dir='output/log', name='SleepVST.kvss_loader')
args = parse_args()

def process_file(hw_path, seq_len, step=10):
    """
    Args:
        hw_path (str): 기준 npy 파일 경로 (hw 파일 경로)
        seq_len (int): 시퀀스 길이
        overlap (int): 오버랩 크기
    Returns:
        list: 처리된 샘플 리스트
    """
    try:
        # 해당 bw 파일 경로 생성
        bw_path = hw_path.replace('_hw.npy', '_bw.npy')
        basename = path.basename(hw_path).replace('_hw.npy', '')
        
        # xml 라벨 파일 경로 생성
        label_path = path.join(args.csv_dir_kvss, f'{basename}_label.csv')
        # 필요한 파일이 모두 존재하는지 확인
        if not path.exists(bw_path) or not path.exists(label_path):
            return []

        # 데이터 로드
        x_hw = np.load(hw_path).astype(np.float32)  # (T, 300)
        x_bw = np.load(bw_path).astype(np.float32)   # (T, 150)
        epochs = parse_csv(label_path)

        if not epochs:
            return []

        labels = np.array([e['label'] for e in epochs])

        T = min(len(x_hw), len(labels), len(x_bw))

        samples = []
        
        for i in range(0, T - seq_len + 1, step):
            samples.append({
                'subject_id': basename,
                'start_idx': i,
                'x_hw': x_hw[i:i + seq_len],
                'x_bw': x_bw[i:i + seq_len],
                'label': labels[i:i + seq_len]
            })
            
        return samples
    except Exception as e:
        logger.error(f"파일 처리 중 오류 {hw_path}: {e}")
        return []

class KVSS(data.Dataset):
    '''
    KVSS 수면 데이터셋
    
    - hw 디렉토리: (T, 300) 형태의 _hw.npy 파일
    - bw 디렉토리: (T, 150) 형태의 _bw.npy 파일
    - xml: 30초 단위 수면 stage label
    
    Args:
        data_dir (str): 기본 데이터 디렉토리 경로
        seq_len (int): 시퀀스 길이 (기본값: 240 에포크 = 2시간)
        split (str): 데이터셋 분할 ('train', 'val', 'test')
        val_ratio (float): 검증 세트 비율 (훈련 세트에서 분할)
        n_jobs (int): 병렬 처리 작업 수
        seed (int): 랜덤 시드
        max_subjects (int, optional): 로드할 최대 피험자 수, None이면 모두 로드
        max_samples (int, optional): 로드할 최대 샘플 수, None이면 모두 로드
    '''
    def __init__(self, 
                 data_dir=args.kvss_npy_dir, 
                 seq_len=240,
                 split='train',
                 val_ratio=0.15,
                 test_ratio=0.15,
                 n_jobs=4,
                 seed=42,):
        self.seq_len = seq_len
        self.split = split
        
        self.exceptions = [
            # [1. edf 파일이 존재하지 않는 case]
            'A2019-EM-01-0119',
            'A2019-EM-01-0120',
            'A2019-EM-01-0122',
            'A2019-EM-01-0123',
            'A2019-EM-01-0124',
            'A2019-EM-01-0125',
            'A2019-EM-01-0196',
            'A2019-EM-01-0197',
            'A2019-EM-01-0198',
            'A2019-EM-01-0199',
            'A2019-EM-01-0200',
            'A2019-EM-01-0201',
            'A2019-EM-01-0202',
            'A2019-EM-01-0203',
            'A2019-EM-01-0204',
            'A2019-EM-01-0205',
            'A2019-EM-01-0206',
            # records 개수에 관한 RuntimeWarning 발생
            'A2021-EM-01-0163',
            ]
        exceptions = [path.join(data_dir, f'{e}_hw.npy') for e in self.exceptions]
        # 데이터 디렉토리 설정
        self.kvss_npy_dir = data_dir
        
        # 모든 hw 파일 찾기
        all_hw_files = sorted(glob(path.join(self.kvss_npy_dir, '*_hw.npy')))
        all_hw_files = [f for f in all_hw_files if f not in exceptions]
        
        if not all_hw_files:
            raise ValueError(f"디렉토리에서 hw 파일을 찾을 수 없음: {self.kvss_npy_dir}")
        
        # 훈련/검증 분할 (테스트 파일 제외하고 분할)
        random.seed(seed)
        random.shuffle(all_hw_files)
        
        train_size = int(len(all_hw_files) * (1 - val_ratio - test_ratio))
        val_size = int(len(all_hw_files) * val_ratio)
        test_size = int(len(all_hw_files) * test_ratio)
        
        train_files = all_hw_files[:train_size]
        val_files = all_hw_files[train_size:train_size + val_size]
        test_files = all_hw_files[train_size + val_size:train_size + val_size + test_size]
        # 훈련/검증/테스트 파일 리스트
        all_files = train_files + val_files + test_files
        
        # 요청된 분할에 따라 파일 선택
        if split == 'train':
            hw_files = train_files
        elif split == 'val':
            hw_files = val_files
        elif split == 'test':
            hw_files = test_files
        elif split == 'all':
            hw_files = all_hw_files
        else:
            raise ValueError(f"지원되지 않는 분할: {split}. 'train', 'val', 또는 'test'를 사용하세요.")
        
        logger.info(f"{split} 분할 - 파일 수: {len(hw_files)}")
        
        # 데이터 로드 - 순차 처리
        self.samples = []
        
        if split == 'train' or split == 'val':
            for hw_file in tqdm(hw_files, desc=f'데이터 로딩 ({split})'):
                hw_file = path.join(data_dir, hw_file)
                result = process_file(hw_file, self.seq_len, step=10)
                if result:
                    self.samples.extend(result)
        else:
            for hw_file in tqdm(hw_files, desc=f'데이터 로딩 ({split})'):
                basename = path.basename(hw_file).replace('_hw.npy', '')
                bw_file = hw_file.replace('_hw.npy', '_bw.npy')
                hw = np.load(hw_file).astype(np.float32)
                bw = np.load(bw_file).astype(np.float32)
                label_path = path.join(args.csv_dir_kvss, f'{basename}_label.csv')
                epochs = parse_csv(label_path)
                
                labels = np.array([e['label'] for e in epochs], dtype=np.int64)
                T = min(len(hw), len(bw), len(labels))
                
                self.samples.append({
                    'subject_id': basename,
                    'start_idx': 0,
                    'x_hw': hw[:T],
                    'x_bw': bw[:T],
                    'label': labels[:T]
                })
                    
        logger.info(f"{split} 데이터셋 준비 완료 - 샘플 수: {len(self.samples)}")

            
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            'x_hw': torch.tensor(sample['x_hw'], dtype=torch.float32),
            'x_bw': torch.tensor(sample['x_bw'], dtype=torch.float32),
            'label': torch.tensor(sample['label'], dtype=torch.long),
            'subject_id': sample['subject_id'],
            'start_idx': sample['start_idx']
        }
    
    def get_class_weights(self):
        all_labels = np.concatenate([sample['label'] for sample in self.samples])
        classes, counts = np.unique(all_labels, return_counts=True)
        weights = 1.0 / counts
        weights = weights / weights.sum() * len(classes)  # 정규화
        return torch.tensor(weights, dtype=torch.float32)
    
    def get_subject_ids(self):
        return sorted(list(set(sample['subject_id'] for sample in self.samples)))
    
    def get_stats(self):
        n_samples = len(self.samples)
        n_subjects = len(self.get_subject_ids())
        
        all_labels = np.concatenate([sample['label'] for sample in self.samples])
        classes, counts = np.unique(all_labels, return_counts=True)
        class_dist = {int(cls): int(cnt) for cls, cnt in zip(classes, counts)}
        
        return {
            "split": self.split,
            "n_samples": n_samples,
            "n_subjects": n_subjects,
            "sequence_length": self.seq_len,
            "total_epochs": len(all_labels),
            "class_distribution": class_dist
        }
        
    def collate_fn(batch, pad_value=0, stack_labels=True, return_dict=True):
        """
        KVSS 데이터셋의 배치를 처리하는 collate 함수입니다.
        
        Args:
            batch (list): 샘플 딕셔너리 리스트
            pad_value (float): 패딩에 사용할 값
            stack_labels (bool): 레이블을 스택할지 여부 (False면 리스트로 유지)
            return_dict (bool): 딕셔너리 형태로 반환할지 여부
            
        Returns:
            dict or tuple: 배치 처리된 텐서
        """
        # 길이가 다를 수 있으므로 각 샘플에서 텐서 추출
        x_hw = [item['x_hw'] for item in batch]
        x_bw = [item['x_bw'] for item in batch]
        labels = [item['label'] for item in batch]
        subject_ids = [item['subject_id'] for item in batch]
        start_idxs = [item['start_idx'] for item in batch]
        
        # 배치 내 최대 길이
        max_len_hw = max(x.shape[0] for x in x_hw)
        max_len_bw = max(x.shape[0] for x in x_bw)
        max_len_label = max(x.shape[0] for x in labels)
        
        # 패딩 적용
        padded_x_hw = []
        padded_x_bw = []
        padded_labels = []
        
        for i in range(len(batch)):
            hw = x_hw[i]
            bw = x_bw[i]
            label = labels[i]
            
            # 각 텐서의 길이 계산
            len_hw = hw.shape[0]
            len_bw = bw.shape[0]
            len_label = label.shape[0]
            
            # 필요한 패딩 길이 계산
            pad_hw = max_len_hw - len_hw
            pad_bw = max_len_bw - len_bw
            pad_label = max_len_label - len_label
            
            # 패딩 적용 (뒤에 패딩 추가)
            if pad_hw > 0:
                hw_pad = torch.full((pad_hw, hw.shape[1]), pad_value, dtype=hw.dtype, device=hw.device)
                hw = torch.cat([hw, hw_pad], dim=0)
                
            if pad_bw > 0:
                bw_pad = torch.full((pad_bw, bw.shape[1]), pad_value, dtype=bw.dtype, device=bw.device)
                bw = torch.cat([bw, bw_pad], dim=0)
                
            if pad_label > 0 and stack_labels:
                label_pad = torch.full((pad_label,), -100, dtype=label.dtype, device=label.device)
                label = torch.cat([label, label_pad], dim=0)
            
            padded_x_hw.append(hw)
            padded_x_bw.append(bw)
            padded_labels.append(label)
        
        # 배치 차원으로 스택
        x_hw_batch = torch.stack(padded_x_hw, dim=0)
        x_bw_batch = torch.stack(padded_x_bw, dim=0)
        
        # 레이블을 스택할지 여부 결정
        if stack_labels:
            labels_batch = torch.stack(padded_labels, dim=0)
        else:
            labels_batch = padded_labels
        
        # 길이 정보 추가 (패딩 마스크 생성 등에 활용 가능)
        lengths = torch.tensor([x.shape[0] for x in x_hw])
        
        if return_dict:
            # 딕셔너리 형태로 반환
            return {
                'x_hw': x_hw_batch,
                'x_bw': x_bw_batch,
                'label': labels_batch,
                'lengths': lengths,
                'subject_ids': subject_ids,
                'start_idxs': start_idxs
            }
        else:
            # 튜플 형태로 반환
            return x_hw_batch, x_bw_batch, labels_batch, lengths, subject_ids, start_idxs

    @staticmethod
    def collate_fn(batch, pad_value=0, stack_labels=True, return_dict=True):
        """
        KVSS 데이터셋의 배치를 처리하는 collate 함수입니다.
        데이터로더에서 사용할 수 있습니다.
        
        사용 예:
            dataloader = DataLoader(
                dataset, 
                batch_size=32, 
                collate_fn=KVSS.collate_fn
            )
        
        커스텀 설정 예:
            dataloader = DataLoader(
                dataset, 
                batch_size=32, 
                collate_fn=lambda batch: KVSS.collate_fn(
                    batch, 
                    pad_value=0, 
                    stack_labels=False
                )
            )
        """
        # 길이가 다를 수 있으므로 각 샘플에서 텐서 추출
        x_hw = [item['x_hw'] for item in batch]
        x_bw = [item['x_bw'] for item in batch]
        labels = [item['label'] for item in batch]
        subject_ids = [item['subject_id'] for item in batch]
        start_idxs = [item['start_idx'] for item in batch]
        
        # 배치 내 최대 길이 구하기
        max_len_hw = max(x.shape[0] for x in x_hw)
        max_len_bw = max(x.shape[0] for x in x_bw)
        max_len_label = max(x.shape[0] for x in labels)
        
        # 패딩 적용
        padded_x_hw = []
        padded_x_bw = []
        padded_labels = []
        
        for i in range(len(batch)):
            hw = x_hw[i]
            bw = x_bw[i]
            label = labels[i]
            
            # 각 텐서의 길이 계산
            len_hw = hw.shape[0]
            len_bw = bw.shape[0]
            len_label = label.shape[0]
            
            # 필요한 패딩 길이 계산
            pad_hw = max_len_hw - len_hw
            pad_bw = max_len_bw - len_bw
            pad_label = max_len_label - len_label
            
            # 패딩 적용 (뒤에 패딩 추가)
            if pad_hw > 0:
                hw_pad = torch.full((pad_hw, hw.shape[1]), pad_value, dtype=hw.dtype, device=hw.device)
                hw = torch.cat([hw, hw_pad], dim=0)
                
            if pad_bw > 0:
                bw_pad = torch.full((pad_bw, bw.shape[1]), pad_value, dtype=bw.dtype, device=bw.device)
                bw = torch.cat([bw, bw_pad], dim=0)
                
            if pad_label > 0 and stack_labels:
                label_pad = torch.full((pad_label,), -100, dtype=label.dtype, device=label.device)
                label = torch.cat([label, label_pad], dim=0)
            
            padded_x_hw.append(hw)
            padded_x_bw.append(bw)
            padded_labels.append(label)
        
        # 배치 차원으로 스택
        x_hw_batch = torch.stack(padded_x_hw, dim=0)
        x_bw_batch = torch.stack(padded_x_bw, dim=0)
        
        # 레이블을 스택할지 여부 결정
        if stack_labels:
            labels_batch = torch.stack(padded_labels, dim=0)
        else:
            labels_batch = padded_labels
        
        # 길이 정보 추가 (패딩 마스크 생성 등에 활용 가능)
        lengths = torch.tensor([x.shape[0] for x in x_hw])
        
        if return_dict:
            # 딕셔너리 형태로 반환
            return {
                'x_hw': x_hw_batch,
                'x_bw': x_bw_batch,
                'label': labels_batch,
                'lengths': lengths,
                'subject_ids': subject_ids,
                'start_idxs': start_idxs
            }
        else:
            # 튜플 형태로 반환
            return x_hw_batch, x_bw_batch, labels_batch, lengths, subject_ids, start_idxs
        
    @classmethod         
    def create_subset(cls, full_dataset, max_samples=None, max_subjects=None, seed=42):
        """
        기존 데이터셋의 서브셋을 생성합니다.
        
        Args:
            full_dataset: 원본 KVSS 데이터셋 객체
            max_samples (int, optional): 최대 샘플 수
            max_subjects (int, optional): 최대 피험자 수
            seed (int): 랜덤 시드
            
        Returns:
            KVSS: 서브셋 데이터셋
        """
        subset = cls.__new__(cls)  # 새 객체 생성 (초기화 없이)
        
        # 필요한 속성 복사
        subset.seq_len = full_dataset.seq_len
        subset.split = full_dataset.split
        subset.test_idx = full_dataset.test_idx
        subset.kvss_npy_dir = full_dataset.kvss_npy_dir
        
        # 새로운 제한 설정
        subset.max_subjects = max_subjects
        subset.max_samples = max_samples
        
        if max_subjects is not None:
            # 고유 피험자 ID 추출
            subject_ids = list(set(sample['subject_id'] for sample in full_dataset.samples))
            
            # 랜덤하게 max_subjects 개수만큼 선택
            random.seed(seed)
            random.shuffle(subject_ids)
            selected_subjects = subject_ids[:max_subjects]
            
            # 선택된 피험자의 샘플만 포함
            subset.samples = [sample for sample in full_dataset.samples 
                             if sample['subject_id'] in selected_subjects]
        else:
            # 피험자 제한 없이 모든 샘플 복사
            subset.samples = full_dataset.samples.copy()
        
        # max_samples 제한 적용
        if max_samples is not None:
            random.seed(seed)
            if len(subset.samples) > max_samples:
                subset.samples = random.sample(subset.samples, max_samples)
        
        logger.info(f"서브셋 생성 완료 - 샘플 수: {len(subset.samples)}, "
                   f"피험자 수: {len(set(sample['subject_id'] for sample in subset.samples))}")
        
        return subset