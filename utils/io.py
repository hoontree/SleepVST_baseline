import os
import numpy as np
import xml.etree.ElementTree as ET
import config

args = config.parse_args()

def check_file_processed(base_name, save_path):
    hw_file = os.path.join(save_path, base_name + '_hw.npy')
    bw_file = os.path.join(save_path, base_name + '_bw.npy')
    
    hw_exists = os.path.exists(hw_file)
    bw_exists = os.path.exists(bw_file)
    
    # 두 파일 모두 존재하면 완전히 처리된 것으로 간주
    if hw_exists:
        # 파일 크기 확인으로 정상 처리 여부 검증
        hw_size = os.path.getsize(hw_file)
        
        if hw_size > 100:  # 최소한의 유효한 데이터 크기 (바이트)
            return True, "complete"
        else:
            return False, "corrupted"  # 파일은 존재하지만 손상되었거나 불완전함
    
    # 일부만 처리된 경우 (불완전한 처리)
    elif bw_exists:
        return False, "partial"
    
    # 처리되지 않은 경우
    else:
        return False, "new"

def sum_stage_wake_duration(xml_path):
    """XML 파일에서 Stages 이벤트의 총 지속 시간(초)을 계산합니다."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    total_duration = 0.0
    for scored_event in root.findall(".//ScoredEvent"):
        event_type = scored_event.find("EventType")
        if event_type is not None and event_type.text is not None and "Stages" in event_type.text:
            duration_elem = scored_event.find("Duration")
            if duration_elem is not None:
                try:
                    total_duration += float(duration_elem.text)
                except ValueError:
                    pass  # 숫자가 아닌 경우 무시
    return total_duration

def get_xml_path_from_edf(edf_path, dataset_type='SHHS'):
    """EDF 파일 경로로부터 해당하는 XML 파일 경로를 생성합니다."""
    base = os.path.splitext(os.path.basename(edf_path))[0]
    xml_filename = f"{base}-nsrr.xml"
    
    if dataset_type.upper() == 'SHHS':
        # edf_path에서 하위 디렉토리(shhs1, shhs2) 추출
        edf_dir = os.path.dirname(edf_path)
        # 마지막 디렉토리 이름 추출 (shhs1 또는 shhs2)
        subdir = os.path.basename(edf_dir)
        
        # XML 디렉토리도 동일한 하위 디렉토리 구조를 가짐
        xml_dir = os.path.join(args.xml_dir_shhs, subdir)
    else:  # MESA
        # MESA 데이터셋의 XML 파일 경로
        xml_dir = config.args.xml_dir_mesa
    
    xml_path = os.path.join(xml_dir, xml_filename)
    return xml_path

def sum_stage_wake_duration_from_csv(csv_path):
    """CSV 파일에서 Stages 이벤트의 총 지속 시간(초)을 계산합니다."""
    data = np.genfromtxt(csv_path, delimiter=',', skip_header=1, dtype=str)
    
    # Stages 열 추출
    stages_column = data[:, 1]  # 두 번째 열이 Stages 열이라고 가정
    total_duration = 0.0
    
    for stage in stages_column:
        if stage == 'Wake':
            total_duration += 1.0  # Wake 상태의 지속 시간 추가 (1초 단위로 가정)
    
    return total_duration