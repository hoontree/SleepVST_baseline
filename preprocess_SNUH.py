import os
import os.path as path
import numpy as np
import gc
import sys
import psutil
import multiprocessing
import mne
import concurrent.futures
import warnings
import signal as sig
import csv
import traceback  # traceback 모듈 추가

from concurrent.futures import ProcessPoolExecutor
from utils.customlogger import Logger
from utils.io import *
from utils.utils_data import *
from mne.io import read_raw_edf
from glob import glob
from tqdm import tqdm


# 전역 변수로 executor 선언
executor = None
args = config.parse_args()
logger = Logger(dir=args.preprocess_log_dir, name='SleepVST.preprocess', run_name=args.run_name)

def signal_handler(sig, frame):
    """
    Ctrl+C 등의 시그널을 처리하는 핸들러
    """
    print("\n프로그램 종료 요청됨. 실행 중인 작업 종료 중...")
    if executor:
        executor.shutdown(wait=False)
    sys.exit(0)

def get_memory_info():
    """
    시스템의 메모리 사용량 정보를 반환
    """
    memory = psutil.virtual_memory()
    return {
        'total': memory.total,
        'available': memory.available,
        'used': memory.used,
        'percent': memory.percent
    }
def get_last_row_column_value(file_path, column_name):
    with open(file_path, 'r', newline='') as f:
        f.seek(0, 2)  # 파일 끝으로 이동
        file_size = f.tell()

        buffer = bytearray()
        pointer = file_size - 1

        # 역방향으로 한 줄 읽기
        while pointer >= 0:
            f.seek(pointer)
            byte = f.read(1)
            if byte == '\n' and buffer:
                break
            buffer.insert(0, ord(byte))
            pointer -= 1

        last_line = buffer.decode('utf-8')

        # 첫 줄(헤더) 읽기
        f.seek(0)
        reader = csv.reader(f)
        header = next(reader)
        col_index = header.index(column_name)

        # 마지막 줄 파싱
        last_values = list(csv.reader([last_line]))[0]  # next() 대신 list()[0] 사용
        return last_values[col_index]
    
def extract_signal(edf_path, channels=['EKG', 'Thorax']):
    """
    Extracts ECG and THOR RES signals from an EDF file.
    Dataset에 따라 다른 채널명과 전처리 방식 적용
    메모리 관리 최적화
    XML 파일에서 계산한 duration까지만 신호 추출
    """
    ann_path = '/tf/00_AIoT2/video_signal/#_2021_Sleep_Video/30sec_labels'
    
    # XML에서 duration 계산
    try:
        basename = os.path.basename(edf_path)
        ann_path = path.join(ann_path, basename.replace('.edf', '_label.csv'))
        if os.path.exists(ann_path):
            duration_sec = int(get_last_row_column_value(ann_path, 'Start_Epoch')) * 30
            if duration_sec <= 0:
                # 유효한 duration이 없으면 전체 파일 사용
                duration_sec = None
                logger.warning(f"Warning: {os.path.basename(edf_path)}의 XML 파일에서 유효한 duration을 찾을 수 없습니다.")
        else:
            duration_sec = None
            logger.warning(f"Warning: {os.path.basename(edf_path)}의 XML 파일을 찾을 수 없습니다: {ann_path}")
    except Exception as e:
        duration_sec = None
        logger.error(f"Error: {os.path.basename(edf_path)}의 XML 파일 처리 중 오류 발생: {str(e)}")
    
    # 기존 mne 설정 임시 저장
    original_verbose = mne.set_log_level('ERROR')
    
    # 표준 출력을 일시적으로 리다이렉트하여 mne 출력 숨기기
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            # 메모리 효율성을 위해 필요한 채널만 로드
            # verbose=False로 설정하여 출력 최소화
            raw = read_raw_edf(edf_path, preload=False, verbose=False)
            
            # 사용 가능한 채널 확인
            available_channels = raw.ch_names
            
            # 요청된 채널이 있는지 확인
            channels_to_pick = []
            channel_map = {}  # 채널 맵핑을 위한 딕셔너리
            
            for i, ch in enumerate(channels):
                if ch in available_channels:
                    channels_to_pick.append(ch)
                    channel_map[ch] = i  # 채널과 인덱스 매핑
            
            if len(channels_to_pick) < 1:
                error_msg = f"필요한 채널을 찾을 수 없습니다: {edf_path}에서 {channels}를 찾을 수 없습니다."
                raise ValueError(error_msg)
            
            # 필요한 채널만 선택
            raw.pick(channels_to_pick)
            
            # 로드 이후에 채널 정보 저장
            channel_names = raw.ch_names  # 선택된 채널 이름 목록
            
            # 이제 필요한 채널만 로드 (verbose=False로 출력 비활성화)
            raw.load_data(verbose=False)
            
            # duration_sec이 유효한 경우 0초부터 duration_sec까지의 데이터만 추출
            if duration_sec is not None and duration_sec > 0:
                # 실제 신호 길이 확인 (초 단위)
                actual_duration = raw.times[-1]
                
                # duration_sec이 실제 신호 길이를 초과하면 실제 길이로 제한
                if duration_sec < actual_duration:
                    # duration_sec이 유효한 범위 내에 있으면 crop 수행
                    raw.crop(tmin=0, tmax=duration_sec, include_tmax=False)
            
            data, _ = raw[:]
            fs = raw.info['sfreq']
            
            # 이제 raw 객체에서 메모리 해제
            del raw
        except Exception as e:
            raise
        finally:
            # 원래 mne 설정 복원
            mne.set_log_level(original_verbose)
    
    try:
        if len(data) < 2:
            raise ValueError(f"데이터셋에는 EKG와 Thorax 두 채널이 필요합니다. 발견된 채널: {channels_to_pick}")
        
        # 채널 맵을 사용하여 올바른 데이터 할당
        ecg_idx = channel_names.index('EKG') if 'EKG' in channel_names else None
        resp_idx = channel_names.index('Thorax') if 'Thorax' in channel_names else None
        
        if ecg_idx is None or resp_idx is None:
            raise ValueError(f"필요한 채널이 없습니다. 발견된 채널: {channel_names}")
        
        ecg_signal = data[ecg_idx]  # EKG
        resp_signal = data[resp_idx]  # Thor
        
        # 메모리 관리: 불필요한 데이터 제거
        del data
        
        hw = preprocess_hw(ecg_signal, fs)
        # 메모리 관리
        del ecg_signal
        
        bw = preprocess_bw(resp_signal, fs)
        # 메모리 관리
        del resp_signal
        
        hw = patchify(hw, patch_size=300, step=300)
        bw = patchify(bw, patch_size=150, step=150)
        return hw, bw
    
    except Exception as e:
        # 명시적 오류 메시지 추가
        raise type(e)(f"{str(e)} (파일: {edf_path}, 채널: {channels_to_pick})")

def process_file(edf_file, save_dir):
    """단일 EDF 파일 처리 함수 (병렬 프로세스 내부)"""
    base = os.path.splitext(os.path.basename(edf_file))[0]

    # 키보드 인터럽트 및 시스템 종료 시그널 무시 (자식 프로세스)
    sig.signal(sig.SIGINT, sig.SIG_IGN)
    sig.signal(sig.SIGTERM, sig.SIG_IGN)

    try:
        processed, status = check_file_processed(base, save_dir)
        if processed:
            return True, "skipped"
        elif status == "partial":
            # 부분적으로 처리된 파일 삭제
            try:
                hw_file = os.path.join(save_dir, base + '_hw.npy')
                bw_file = os.path.join(save_dir, base + '_bw.npy')
                if os.path.exists(hw_file):
                    os.remove(hw_file)
                if os.path.exists(bw_file):
                    os.remove(bw_file)
            except Exception as e:
                return False, f"error_removing_partial: {str(e)}"

        # 메모리 사용량 최적화
        gc.collect()
        channels = ['EKG', 'Thorax']
        
        # 신호 추출 및 처리
        hw, bw = extract_signal(edf_file, channels=channels)

        # 결과 저장
        np.save(os.path.join(save_dir, base + '_hw.npy'), hw)
        np.save(os.path.join(save_dir, base + '_bw.npy'), bw)

        # 메모리 해제
        del hw, bw
        gc.collect()

        return True, "processed"
    except MemoryError:
        return False, f"error_memory: 메모리 부족 - {base}"
    except Exception as e:
        return False, f"error_processing: {type(e).__name__}: {str(e)}"

def main():    
    
    dataset = 'KISS'
    
    # 시그널 핸들러 등록
    sig.signal(sig.SIGINT, signal_handler)
    sig.signal(sig.SIGTERM, signal_handler)
    
    # 메모리 사용 정보 출력
    memory_info = get_memory_info()
    print(f"사용 가능한 메모리: {memory_info['available']/1024/1024:.1f} MB / 총 메모리: {memory_info['total']/1024/1024:.1f} MB")
    logger.info(f"사용 가능한 메모리: {memory_info['available']/1024/1024:.1f} MB / 총 메모리: {memory_info['total']/1024/1024:.1f} MB")

    logger.info("전처리 작업 시작")
    logger.info(f"설정: {args}")

    global executor
    
    edf_dir = '/tf/00_AIoT2/video_signal/#_2021_Sleep_Video/Processed'
    save_dir = os.path.join(args.save_dir, 'kiss')       
    os.makedirs(save_dir, exist_ok=True)

    edf_files = glob(os.path.join(edf_dir, 'A*.edf'))
    logger.info(f'데이터셋 {dataset} 처리 시작 (총 {len(edf_files)} 파일)')

    # 이미 처리된 파일 확인 (메인 프로세스에서만)
    complete_count = 0
    for edf_file in edf_files:
        base = path.splitext(path.basename(edf_file))[0]
        processed, _ = check_file_processed(base, save_dir)
        if processed:
            complete_count += 1
    print(f"이미 처리된 파일: {complete_count}/{len(edf_files)} ({complete_count/len(edf_files)*100:.1f}%)")
    logger.info(f"이미 처리된 파일: {complete_count}/{len(edf_files)} ({complete_count/len(edf_files)*100:.1f}%)")

    # 처리할 파일 필터링 (이미 처리된 파일 제외)
    files_to_process = []
    for edf_file in edf_files:
        base = path.splitext(path.basename(edf_file))[0]
        processed, _ = check_file_processed(base, save_dir)
        if not processed:
            files_to_process.append(edf_file)
    
    print(f"처리할 파일: {len(files_to_process)} 개")
    
    # 전체 결과 집계
    overall_results = {"processed": 0, "skipped": 0, "error": 0}
    
    # 배치 크기 설정: CPU 코어 수의 2~4배가 일반적으로 적당함
    # 메모리 문제가 있을 경우 이 값을 줄임
    batch_size = max(1, min(multiprocessing.cpu_count() * 2, 8))
    num_workers = max(1, min(multiprocessing.cpu_count() - 1, 4))  # 워커 수 제한
    
    print(f"배치 크기: {batch_size}, 워커 수: {num_workers}")
    logger.info(f"배치 크기: {batch_size}, 워커 수: {num_workers}")

    # 배치 단위로 파일 처리
    batches = [files_to_process[i:i + batch_size] for i in range(0, len(files_to_process), batch_size)]
    
    # 전체 진행 상황 표시
    with tqdm(total=len(files_to_process), ncols=100, desc=f"{dataset}", unit="file") as pbar:
        for batch_idx, batch in enumerate(batches):
            try:
                # 워커 풀 초기화 (각 배치마다 새로 생성)
                try:
                    # spawn 방식 사용 (fork 대신) - 멀티프로세싱 관련 문제 예방
                    multiprocessing.set_start_method('spawn', force=True)
                except RuntimeError:
                    # 이미 설정된 경우 무시
                    pass
                
                with ProcessPoolExecutor(max_workers=num_workers) as executor_local:
                    executor = executor_local
                    
                    # 배치 작업 제출
                    futures = {
                        executor.submit(
                            process_file, 
                            edf_file, 
                            save_dir
                        ): path.basename(edf_file) 
                        for edf_file in batch
                    }
                    
                    # 배치 결과 집계
                    batch_results = {"processed": 0, "skipped": 0, "error": 0}
                    
                    # 타임아웃 설정을 통한 처리
                    for future in concurrent.futures.as_completed(futures):
                        file_name = futures[future]
                        pbar.update(1)
                        
                        try:
                            # 타임아웃 설정 (600초 = 10분)
                            success, status = future.result(timeout=600)
                            
                            if status.startswith("error"):
                                batch_results["error"] += 1
                                # 오류는 메인 프로세스에서 로깅
                                error_msg = f"{file_name}: {status}"
                                logger.error(error_msg)
                            elif status == "skipped":
                                batch_results["skipped"] += 1
                            elif status == "processed":
                                batch_results["processed"] += 1
                                logger.debug(f"Processed: {file_name}")
                            
                            pbar.set_postfix(
                                processed=overall_results["processed"] + batch_results["processed"], 
                                skipped=overall_results["skipped"] + batch_results["skipped"], 
                                error=overall_results["error"] + batch_results["error"]
                            )
                        
                        except concurrent.futures.TimeoutError:
                            batch_results["error"] += 1
                            logger.error(f"{file_name} 처리 시간 초과 (10분)")
                        
                        except Exception as e:
                            batch_results["error"] += 1
                            logger.error(f"{file_name} 처리 중 예상치 못한 오류: {traceback.format_exc()}")
                
                # 전체 결과에 배치 결과 추가
                overall_results["processed"] += batch_results["processed"]
                overall_results["skipped"] += batch_results["skipped"]
                overall_results["error"] += batch_results["error"]
                
                # 배치 처리 후 명시적 가비지 컬렉션
                gc.collect()
                
            except KeyboardInterrupt:
                logger.warning("배치 처리 중단됨. 다음 배치로 진행합니다.")
                print("\n배치 처리 중단됨. 다음 배치로 진행합니다.")
                continue
                
            except Exception as e:
                logger.error(f"배치 {batch_idx+1}/{len(batches)} 처리 중 오류 발생: {str(e)}")
                continue
            
            finally:
                executor = None
        
        # 데이터셋 처리 결과 요약
        summary = f"데이터셋 {dataset} 처리 결과: 처리 완료={overall_results['processed']}, 건너뛴 파일={overall_results['skipped']}, 오류={overall_results['error']}"
        print(summary)
        logger.info(summary)
    
    logger.info(f'데이터셋 {dataset} 처리 완료')
    
    logger.info("모든 데이터셋 처리 완료")

if __name__ == "__main__":
    main()