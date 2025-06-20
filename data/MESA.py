import numpy as np
import os.path as path
import torch
import torch.utils.data as data
import random
import re
from utils.util import parse_xml
from glob import glob
from utils.customlogger import Logger
from config import parse_args
from tqdm import tqdm
from data.utils import collate_fn as shared_collate_fn

# 로깅 설정
logger = Logger(dir='output/log', name='SleepVST.mesa_loader')
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
        label_path = path.join(args.xml_dir_mesa, f'{basename}-nsrr.xml')
        # 필요한 파일이 모두 존재하는지 확인
        if not path.exists(bw_path) or not path.exists(label_path):
            return []

        # 데이터 로드
        x_hw = np.load(hw_path).astype(np.float32)  # (T, 300)
        x_bw = np.load(bw_path).astype(np.float32)   # (T, 150)
        epochs = parse_xml(label_path)

        if not epochs:
            return []

        labels = np.array([e['label'] for e in epochs])

        # 정렬된 길이로 자르기
        T = min(len(x_hw), len(labels), len(x_bw))
        if x_hw.shape[0] != T or x_bw.shape[0] != T or labels.shape[0] != T:
            logger.warning(f"파일 길이 불일치: {hw_path} (hw: {len(x_hw)}, bw: {len(x_bw)}, label: {len(labels)})")

        samples = []
        
        # 시퀀스 길이(예: 240 에포크 = 2시간)로 청킹, 오버랩 지원
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

class MESA(data.Dataset):
    '''
    MESA 수면 데이터셋
    
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
                 data_dir=args.mesa_npy_dir, 
                 seq_len=240,
                 split='train',
                 val_ratio=0.25,
                 n_jobs=4,
                 seed=42,):
        self.seq_len = seq_len
        self.split = split
        self.test_idx = [
            1, 21, 33, 52, 77, 81, 101, 111, 225, 310, 314, 402, 416, 445, 465, 483, 505, 554, 572, 587, 601, 620, 648, 702, 764, 771, 792, 797, 800, 807, 860, 892, 902, 904, 921, 1033, 1080, 1121, 1140, 1148, 1161, 1164, 1219, 1224, 1271, 1324, 1356, 1391, 1463, 1483, 1497, 1528, 1531, 1539, 1672, 1693, 1704, 1874, 1876, 1900, 1914, 2039, 2049, 2096, 2100, 2109, 2169, 2172, 2183, 2208, 2239, 2243, 2260, 2269, 2317, 2362, 2388, 2470, 2472, 2488, 2527, 2556, 2602, 2608, 2613, 2677, 2680, 2685, 2727, 2729, 2802, 2811, 2828, 2877, 2881, 2932, 2934, 2993, 2999, 3044, 3066, 3068, 3111, 3121, 3153, 3275, 3298, 3324, 3369, 3492, 3543, 3554, 3557, 3561, 3684, 3689, 3777, 3793, 3801, 3815, 3839, 3886, 3997, 4110, 4137, 4171, 4227, 4285, 4332, 4406, 4460, 4462, 4497, 4501, 4552, 4577, 4649, 4650, 4667, 4732, 4794, 4888, 4892, 4895, 4912, 4918, 4998, 5006, 5075, 5077, 5148, 5169, 5203, 5232, 5243, 5287, 5316, 5357, 5366, 5395, 5397, 5457, 5472, 5479, 5496, 5532, 5568, 5580, 5659, 5692, 5706, 5737, 5754, 5805, 5838, 5847, 5890, 5909, 5957, 5983, 6015, 6039, 6047, 6123, 6224, 6263, 6266, 6281, 6291, 6482, 6491, 6502, 6516, 6566, 6567, 6583, 6619, 6629, 6646, 6680, 6722, 6730, 6741, 6788
        ]
        
        # 데이터 디렉토리 설정
        self.mesa_npy_dir = data_dir
        
        # 모든 hw 파일 찾기
        all_hw_files = sorted(glob(path.join(self.mesa_npy_dir, '*_hw.npy')))
        
        if not all_hw_files:
            raise ValueError(f"디렉토리에서 hw 파일을 찾을 수 없음: {self.mesa_npy_dir}")
        
        # 테스트 인덱스 목록 사용하여 테스트 파일 분리
        test_files = []
        non_test_files = []
        
        # 파일명에서 ID 추출하여 테스트 셋과 비교
        for hw_file in all_hw_files:
            basename = path.basename(hw_file)
            match = re.search(r'(\d{4})', basename)
            if match:
                file_id = int(match.group(1))

                if file_id in self.test_idx:
                    test_files.append(hw_file)
                else:
                    non_test_files.append(hw_file)
        
        logger.info(f"테스트 파일 수: {len(test_files)}, 비테스트 파일 수: {len(non_test_files)}")
        
        # 훈련/검증 분할 (테스트 파일 제외하고 분할)
        random.seed(seed)
        random.shuffle(non_test_files)
        
        val_size = int(len(non_test_files) * val_ratio)
        train_files = non_test_files[val_size:]
        val_files = non_test_files[:val_size]
        
        # 요청된 분할에 따라 파일 선택
        if split == 'train':
            hw_files = train_files
        elif split == 'val':
            hw_files = val_files
        elif split == 'test':
            hw_files = test_files
        else:
            raise ValueError(f"지원되지 않는 분할: {split}. 'train', 'val', 또는 'test'를 사용하세요.")
        
        logger.info(f"{split} 분할 - 파일 수: {len(hw_files)}")
        
        # 데이터 로드 - 순차 처리
        self.samples = []
        
        if split == 'train' or split == 'val':
            for hw_file in tqdm(hw_files, desc=f'데이터 로딩 ({split})'):
                hw_file = path.join(self.mesa_npy_dir, hw_file)
                result = process_file(hw_file, self.seq_len, step=10)
                if result:
                    self.samples.extend(result)
                    if self.max_samples is not None and len(self.samples) >= self.max_samples:
                        self.samples = self.samples[:self.max_samples]
                        logger.info(f"최대 샘플 수 {self.max_samples}개 도달, 데이터 로딩 중단")
                        break
        else:
            for hw_file in tqdm(hw_files, desc=f'데이터 로딩 ({split})'):
                bw_file = hw_file.replace('_hw.npy', '_bw.npy')
                basename = path.basename(hw_file).replace('_hw.npy', '')
                hw = np.load(hw_file).astype(np.float32)
                bw = np.load(bw_file).astype(np.float32)
                label_path = path.join(args.xml_dir_mesa, f'{basename}-nsrr.xml')
                epochs = parse_xml(label_path)
                labels = np.array([e['label'] for e in epochs])
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
        

    @staticmethod
    def collate_fn(batch, pad_value=0, stack_labels=True, return_dict=True):
        """Wrapper around :func:`data.utils.collate_fn`."""
        return shared_collate_fn(batch, pad_value=pad_value,
                                stack_labels=stack_labels,
                                return_dict=return_dict)
        
    @classmethod         
    def create_subset(cls, full_dataset, max_samples=None, max_subjects=None, seed=42):
        """
        기존 데이터셋의 서브셋을 생성합니다.
        
        Args:
            full_dataset: 원본 MESA 데이터셋 객체
            max_samples (int, optional): 최대 샘플 수
            max_subjects (int, optional): 최대 피험자 수
            seed (int): 랜덤 시드
            
        Returns:
            MESA: 서브셋 데이터셋
        """
        subset = cls.__new__(cls)  # 새 객체 생성 (초기화 없이)
        
        # 필요한 속성 복사
        subset.seq_len = full_dataset.seq_len
        subset.split = full_dataset.split
        subset.test_idx = full_dataset.test_idx
        subset.mesa_npy_dir = full_dataset.mesa_npy_dir
        
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