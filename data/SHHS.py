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

logger = Logger(dir='output/log', name='SleepVST.shhs_loader')
args = parse_args()

def process_file(hw_path, seq_len, step=10):
    """
    Args:
        hw_path (str): 기준 npy 파일 경로 (hw 파일 경로)
        seq_len (int): 시퀀스 길이 (기본값: 240 epochs, 2h)
        step (int): 시퀀스 추출 간격 (default: 10, 5m)
    Returns:
        list: 처리된 샘플 리스트
    """
    try:        
        # 해당 bw 파일 경로 생성
        bw_path = hw_path.replace('_hw.npy', '_bw.npy')
        basename = path.basename(hw_path).replace('_hw.npy', '')
        
        # xml 라벨 파일 경로 생성
        if 'shhs1' in hw_path:
            label_path = path.join(args.xml_dir_shhs, 'shhs1', f'{basename}-nsrr.xml')
        elif 'shhs2' in hw_path:
            label_path = path.join(args.xml_dir_shhs, 'shhs2', f'{basename}-nsrr.xml')
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
            x_hw = x_hw[:T]
            x_bw = x_bw[:T]
            labels = labels[:T]


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

class SHHS(data.Dataset):
    '''
    SHHS 수면 데이터셋
    
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
    '''
    def __init__(self, 
                 data_dir=args.shhs_npy_dir, 
                 seq_len=240,
                 split='train',
                 val_ratio=0.25,      # 훈련에서 검증 분할 비율
                 n_jobs=4,
                 seed=42,
                 max_subjects=None,
                 max_samples=None):
        self.seq_len = seq_len
        self.split = split
        self.max_subjects = max_subjects
        self.max_samples = max_samples
        self.test_idx = [
            203949, 204944, 200702, 203956, 202106, 201917, 203231, 204150, 205275, 203034, 201936, 205025, 200885, 201204, 204594, 201308, 204960, 205608, 204379, 203354, 204125, 204330, 203384, 201213, 201598, 204495, 204590, 203944, 203945, 202921, 201287, 203495, 205462, 204068, 203423, 202981, 203505, 204079, 204939, 203390, 204179, 204885, 202435, 202157, 202834, 200825, 203684, 205299, 200897, 200152, 202828, 203530, 203312, 205398, 203984, 202801, 203264, 201453, 203652, 200460, 202820, 204638, 203367, 205565, 200839, 203213, 204016, 204473, 200751, 201206, 200927, 201608, 201102, 205494, 201399, 200730, 202948, 200293, 204865, 203520, 204795, 200953, 203125, 205340, 205009, 204825, 200752, 202794, 203165, 203306, 203490, 204041, 202383, 204656, 203772, 203829, 204517, 201557, 202650, 203308, 203564, 202210, 200991, 205663, 203559, 204303, 200513, 205664, 202152, 204928, 203974, 200851, 205169, 205645, 204256, 201783, 201414, 204540, 204661, 203894, 201373, 202496, 200718, 202227, 200243, 203826, 202946, 203446, 202123, 200604, 201058, 205704, 204798, 205346, 204823, 203748, 203824, 200680, 200698, 203166, 204140, 205072, 203512, 200646, 205744, 200679, 204295, 201670, 204486, 203316, 203511, 200998, 204335, 200886, 202968, 204409, 205575, 202943, 205082, 203200, 200321, 205226, 200899, 203845, 202940, 200687, 200948, 203311, 200769, 200782, 203925, 200666, 202480, 201521, 205593, 200507, 203282, 204001, 205601, 203372, 204988, 204269, 203946, 205257, 200176, 204231, 204530, 204963, 200145, 200888, 202187, 200088, 203882, 203286, 204418, 200578, 204273, 200823, 203065, 205349, 203106, 200217, 202912, 203566, 200154, 201323, 200662, 204289, 205651, 203252, 205044, 201024, 203716, 204232, 200632, 200102, 205450, 200566, 202521, 202563, 200584, 203192, 201298, 205485, 205739, 204617, 204642, 201331, 205596, 203202, 203381, 204956, 203522, 200303, 203534, 204190, 201130, 201268, 200191, 201513, 200955, 203689, 203157, 201401, 201517, 202489, 203018, 203232, 205146, 202963, 202821, 204287, 204422, 204472, 200334, 200925, 203135, 200516, 202442, 204171, 203392, 201503, 202458, 205587, 203502, 203626, 204702, 204599, 200150, 205605, 202566, 204296, 202221, 203296, 205537, 203860, 200842, 200318, 204023, 202463, 201628, 200858, 205419, 201068, 205312, 202663, 202444, 200579, 201629, 203198, 204283, 204846, 204699, 200178, 200981, 203237, 200564, 204340, 202785, 202938, 204506, 201493, 203610, 201353, 203769, 200105, 204522, 204343, 204086, 200320, 204425, 202842, 204459, 202226, 203528, 204978, 204461, 204691, 200111, 201083, 200950, 200108, 203455, 204647, 200952, 204443, 204435, 204504, 205064, 203476, 203695, 203721, 202201, 200841, 200935, 204093, 201223, 201470, 204747, 205350, 204871, 203303, 204132, 201219, 204676, 201402, 205722, 204115, 200744, 205772, 203281, 204926, 205086, 202990, 203235, 204384, 201432, 203039, 200387, 203961, 203456, 203462, 202546, 203895, 205595, 204496, 201241, 205004, 200712, 204565, 203671, 200936, 201271, 203734, 202428, 203260, 200653, 204405, 201982, 200703, 203314, 202405, 200668, 205702, 204431, 202608, 203056, 204544, 203754, 205761, 203942, 200835, 200192, 205539, 200945, 203254, 204177, 200219, 205486, 202642, 202417, 203498, 203347, 204759, 202361, 205255, 201064, 201544, 200295, 204233, 204187, 202150, 204224, 204370, 204952, 200571, 203976, 204304, 200347, 205126, 203224, 200591, 203060, 201543, 204923, 201778, 204914, 205222, 204907, 200766, 205305, 204235, 200406, 205661, 200209, 204778, 201640, 204236, 200901, 205356, 200853, 200210, 204898, 203269, 203451, 202825, 201299, 204690, 203557, 203589, 203037, 204337, 204460, 201316, 201312, 204856, 203138, 203412, 200242, 203460, 200920, 200887, 201918, 203395, 205530, 201349, 200829, 203208, 201519, 200386, 203117, 200466, 202605, 203121, 200624, 205721, 204323, 204554, 205289, 204934, 200233, 203149, 204170, 203966, 205252, 205548, 203006, 202902, 203818, 202942, 201018, 205588, 202395, 203709, 205591, 205532, 204763, 202829, 205626, 201538
        ]
        
        # 데이터 디렉토리 설정
        self.shhs_npy_dir = data_dir
        
        # 모든 hw 파일 찾기
        all_hw_files = sorted(glob(path.join(self.shhs_npy_dir, '*_hw.npy')))
        
        if not all_hw_files:
            raise ValueError(f"디렉토리에서 hw 파일을 찾을 수 없음: {self.shhs_npy_dir}")
        
        # 테스트 인덱스 목록 사용하여 테스트 파일 분리
        test_files = []
        non_test_files = []
        
        # 파일명에서 ID 추출하여 테스트 셋과 비교
        for hw_file in all_hw_files:
            basename = path.basename(hw_file)
            match = re.search(r'(\d{6})', basename)
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
        
        # max_subjects 제한 적용
        if self.max_subjects is not None and self.max_subjects > 0:
            hw_files = hw_files[:self.max_subjects]
            logger.info(f"{split} 분할 - 제한된 피험자 수: {len(hw_files)}/{len(train_files if split=='train' else val_files if split=='val' else test_files)}")
        
        logger.info(f"{split} 분할 - 파일 수: {len(hw_files)}")
        
        # 데이터 로드 - 병렬 처리 대신 순차 처리
        self.samples = []
        if self.split == 'train' or self.split == 'val':
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
            for hw_file in tqdm(hw_files, desc=f'테스트 데이터 로딩'):
                basename = path.basename(hw_file).replace('_hw.npy', '')
                hw = np.load(hw_file).astype(np.float32)
                bw = np.load(hw_file.replace('_hw.npy', '_bw.npy')).astype(np.float32)
                if 'shhs1' in basename:
                    label_path = path.join(args.xml_dir_shhs, 'shhs1', f'{basename}-nsrr.xml')
                else:
                    label_path = path.join(args.xml_dir_shhs, 'shhs2', f'{basename}-nsrr.xml')
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
        
    def collate_fn(batch, pad_value=0, stack_labels=True, return_dict=True):
        """
        SHHS 데이터셋의 배치를 처리하는 collate 함수입니다.
        
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

    # SHHS 클래스에 정적 메서드로 collate_fn 추가   
    @staticmethod
    def collate_fn(batch, pad_value=0, stack_labels=True, return_dict=True):
        """
        SHHS 데이터셋의 배치를 처리하는 collate 함수입니다.
        데이터로더에서 사용할 수 있습니다.
        
        사용 예:
            dataloader = DataLoader(
                dataset, 
                batch_size=32, 
                collate_fn=SHHS.collate_fn
            )
        
        커스텀 설정 예:
            dataloader = DataLoader(
                dataset, 
                batch_size=32, 
                collate_fn=lambda batch: SHHS.collate_fn(
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