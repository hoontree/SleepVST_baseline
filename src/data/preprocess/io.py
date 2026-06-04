import os
import numpy as np
import xml.etree.ElementTree as ET
from tqdm import tqdm
import imageio.v3 as iio
import cv2
import skimage.transform as sktransform
import imageio_ffmpeg as ffmpeg
import itertools

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

def get_xml_path_from_edf(edf_path, xml_dir_shhs, xml_dir_mesa, dataset_type='SHHS'):
    """EDF 파일 경로로부터 해당하는 XML 파일 경로를 생성합니다."""
    base = os.path.splitext(os.path.basename(edf_path))[0]
    xml_filename = f"{base}-nsrr.xml"
    
    if dataset_type.upper() == 'SHHS':
        # edf_path에서 하위 디렉토리(shhs1, shhs2) 추출
        edf_dir = os.path.dirname(edf_path)
        # 마지막 디렉토리 이름 추출 (shhs1 또는 shhs2)
        subdir = os.path.basename(edf_dir)
        
        # XML 디렉토리도 동일한 하위 디렉토리 구조를 가짐
        xml_dir = os.path.join(xml_dir_shhs, subdir)
    else:  # MESA
        # MESA 데이터셋의 XML 파일 경로
        xml_dir = xml_dir_mesa
    
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

def loadVideo(file_dir, start_frame=0, end_frame=None):
    '''
    Load RGB video using imageio (uint8)
    - file_dir: video file path (ex> end with ".mp4")
    '''
    reader = iio.imiter(file_dir)
    orig_vid = []
    for i, im in tqdm(enumerate(itertools.islice(reader, start_frame, end_frame)), 
                      ascii=True, 
                      desc="Load Video"):
        orig_vid.append(im)
        
    return np.array(orig_vid)

def loadVideoStream(reader_iter, num_frames):
    '''
    Load RGB video frames from an active iterator
    - reader_iter: active imageio iterator
    - num_frames: number of frames to read
    Returns:
        numpy array of frames or None if stream ended
    '''
    frames = []
    try:
        for _ in range(num_frames):
            frame = next(reader_iter)
            frames.append(frame)
    except StopIteration:
        # Stream ended
        if len(frames) == 0:
            return None
        # Return partial frames if any
        return np.array(frames)
    
    return np.array(frames)

def loadVideoStreamCV2(cap, num_frames, crop_box=None, size=None, normalize=True):
    '''
    Load RGB video frames from an active cv2.VideoCapture object
    - cap: active cv2.VideoCapture object
    - num_frames: number of frames to read
    - crop_box: tuple (y1, y2, x1, x2) for cropping, None to skip cropping
    - size: (width, height) for resizing, None to skip resizing
    - normalize: if True, normalize to [0, 1], otherwise keep as uint8
    Returns:
        numpy array of frames or None if stream ended
    '''
    frames = []
    
    for _ in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            # Stream ended
            if len(frames) == 0:
                return None
            # Return partial frames if any
            break
        
        # Crop if specified
        if crop_box is not None:
            y1, y2, x1, x2 = crop_box
            frame = frame[y1:y2, x1:x2]
        
        # BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize if specified
        if size is not None:
            w, h = size
            frame = cv2.resize(frame, (w, h))
        
        # Normalize if specified
        if normalize:
            frame = frame / 255.0
        
        frames.append(frame)
    
    return np.array(frames)

def FrameResize(video, size):
    '''
    Resize video frames (ndarray)
    - video: array of video frames (num_frames, width, height, channel)
    - size: (width, height)
    '''
    w, h = size
    frames_resized = np.zeros((len(video), w, h, 3))
    for i in tqdm(range(len(video)),
                  ascii=True, 
                  desc="Frame Resizing"):
        frames_resized[i] = sktransform.resize(video[i], size)
        
    return frames_resized

def loadFrames(frames_dir, ratio, size=None):
    '''
    Load RGB video frames (uint8)
    - frames_dir: frames directory path (frames: ".jpg")
    - size: (width, height)
    '''    
    frame_names = [f for f in sorted(os.listdir(frames_dir)) if f.endswith('.jpg')]
    
    num = int(len(frame_names)*ratio)
    interval = int(1 / ratio)
    frames = []
    
    for i in tqdm(range(num), 
                  ascii=True, 
                  desc="Load Frames"):

        frame_file = os.path.join(frames_dir, frame_names[i*interval])
        
        # Read frame
        frame = cv2.imread(frame_file)
        
        # Crop
        frame = frame[32:422, 125:515]
        #frame = frame[52:402, 145:495]
        # BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Resize
        if size != None:
            w, h = size
            frame = cv2.resize(frame, (w, h))
        # Normalize
        frame = frame / 255.
        
        frames.append(frame)

    return np.array(frames)