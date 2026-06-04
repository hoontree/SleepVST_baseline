from glob import glob
import cv2
import numpy as np
from tqdm import tqdm
import argparse
import os
import sys

parser = argparse.ArgumentParser(description="Compute motion features from a video.")

parser.add_argument('--video_path', type=str, default='/tf/00_data/#_2021_Sleep_Video/', help='Path to the input video file.')
parser.add_argument('--target_fps', type=int, default=4, help='Target FPS for optical flow computation.')
parser.add_argument('--out_dir', type=str, default='/tf/01_code/mylittlecodes/SleepVST_baseline/data/motionfeatures', help='Path to the output directory.')
parser.add_argument('--out_size', type=int, nargs=2, default=(224, 224), help='Output size for homography transformation.')


def homography_transform(pts_src, out_size, img):
    """
    Apply a homography transformation to an image based on source and destination points.
    
    Args:
        pts_src (np.ndarray): Source points in the format [[x1, y1], [x2, y2], ...].
        pts_dst (np.ndarray): Destination points in the format [[x1, y1], [x2, y2], ...].
        img (np.ndarray): Input image to be transformed.
    
    Returns:
        np.ndarray: Transformed image.
    """
    pts_dst = np.array([[0, 0], [out_size[0], 0], [out_size[0], out_size[1]], [0, out_size[1]]], dtype=np.float32)
    pts_src = scale_quad(np.array(pts_src, dtype=np.float32), scale=1.2)
    h, status = cv2.findHomography(pts_src, pts_dst)
    return cv2.warpPerspective(img, h, out_size)

def scale_quad(pts: np.ndarray, scale: float = 1.2) -> np.ndarray:
    """
    주어진 사각형(4×2 배열)을 무게중심 기준으로 확대·축소한다.
    
    Args:
        pts   : np.ndarray, shape (4, 2), float32 — 꼭짓점 좌표
        scale : float — 1보다 크면 확대, 1보다 작으면 축소
    
    Returns:
        np.ndarray 같은 shape — 새 꼭짓점 좌표
    """
    center = pts.mean(axis=0)
    return (pts - center) * scale + center  

def extract_region_magnitudes_from_box(flow):
    """
    Optical flow로부터 고정된 person box와 head 영역을 기준으로
    Head, Body, Outer의 max/mean magnitude를 계산합니다.
    """
    h, w = flow.shape[:2]
    mag = np.linalg.norm(flow, axis=2)

    # 사각형 기준
    x0, y0, x1, y1 = 46, 78, 372, 570
    head_y1 = 153  # head: y=78~153
    mask_H = np.zeros((h, w), dtype=np.uint8)
    mask_B = np.zeros((h, w), dtype=np.uint8)
    mask_O = np.ones((h, w), dtype=np.uint8)

    mask_H[y0:head_y1, x0:x1] = 1
    mask_B[head_y1:y1, x0:x1] = 1
    mask_O[y0:y1, x0:x1] = 0  # person box 제외 영역

    def stats(mask):
        vals = mag[mask == 1]
        return (np.max(vals) if vals.size > 0 else 0.0,
                np.mean(vals) if vals.size > 0 else 0.0)

    vH, sH = stats(mask_H)
    vB, sB = stats(mask_B)
    vO, sO = stats(mask_O)
    return {'Head': (vH, sH), 'Body': (vB, sB), 'Outer': (vO, sO)}

def compute_motion_features(v_seq, s_seq, seconds=[30, 300], delta_fps=4, thresholds=[0.01, 0.1, 1.0]):
    """
    Args:
        v_seq (np.ndarray): shape (T,), 각 시점 t에서 v(t;R) = max flow magnitude
        s_seq (np.ndarray): shape (T,), 각 시점 t에서 s(t;R) = mean flow magnitude
        seconds (list): f1, f2 계산을 위한 누적 시간 (초)
        delta_fps (int): optical flow 계산 프레임률 (예: 4Hz)
        thresholds (list): f3, f4 계산을 위한 threshold 리스트

    Returns:
        features (dict): key는 'f1', 'f2', 'f3@0.01', 'f4@0.1' 같은 문자열
                         value는 shape (T,)인 np.ndarray
    """
    T = len(v_seq)
    for delta in seconds:
        delta_frames = int(delta * delta_fps)
        features = {}

        # f1(t) = sum of v(t') over [t - Δ, t]
        f1 = np.array([np.sum(v_seq[max(0, t-delta_frames):t+1]) for t in range(T)])
        features[f'f1@{delta}s'] = f1

        # f2(t) = sum of s(t') over [t - Δ, t]
        f2 = np.array([np.sum(s_seq[max(0, t-delta_frames):t+1]) for t in range(T)])
        features[f'f2@{delta}s'] = f2

    # f3/f4 for each threshold δ
    for delta in thresholds:
        f3 = []
        f4 = []
        last_above_v = -1
        last_above_s = -1

        for t in range(T):
            # v(t') > δ 찾기
            for t_prime in reversed(range(t+1)):
                if v_seq[t_prime] > delta:
                    last_above_v = t_prime
                    break
            # s(t') > δ 찾기
            for t_prime in reversed(range(t+1)):
                if s_seq[t_prime] > delta:
                    last_above_s = t_prime
                    break

            tau_v = last_above_v if last_above_v >= 0 else 0
            tau_s = last_above_s if last_above_s >= 0 else 0

            f3.append(t - tau_v + 1)
            f4.append(t - tau_s + 1)

        features[f'f3@{delta}'] = np.array(f3)
        features[f'f4@{delta}'] = np.array(f4)

    return features
def compute_motion_features_fast(v_seq, s_seq, seconds=[30, 300], delta_fps=4, thresholds=[0.01, 0.1, 1.0]):
    T = len(v_seq)
    features = {}

    # 누적합 기반 f1, f2 (O(T))
    v_cum = np.cumsum(v_seq)
    s_cum = np.cumsum(s_seq)
    for sec in seconds:
        delta = int(sec * delta_fps)
        f1 = np.zeros(T)
        f2 = np.zeros(T)
        for t in range(T):
            if t >= delta:
                f1[t] = v_cum[t] - v_cum[t - delta]
                f2[t] = s_cum[t] - s_cum[t - delta]
            else:
                f1[t] = v_cum[t]
                f2[t] = s_cum[t]
        features[f'f1@{sec}s'] = f1
        features[f'f2@{sec}s'] = f2

    # 마지막 움직임 이후 경과 시간 (O(T))
    for delta in thresholds:
        f3 = np.zeros(T, dtype=np.int32)
        f4 = np.zeros(T, dtype=np.int32)
        last_v = -1
        last_s = -1
        for t in range(T):
            if v_seq[t] > delta:
                last_v = t
            if s_seq[t] > delta:
                last_s = t
            f3[t] = t - last_v + 1 if last_v >= 0 else t + 1
            f4[t] = t - last_s + 1 if last_s >= 0 else t + 1
        features[f'f3@{delta}'] = f3
        features[f'f4@{delta}'] = f4

    return features

def compute_optical_flow(prev_img, curr_img, flow_calculator=None):
    """
    Compute optical flow between two images using Farneback method.
    
    Args:
        prev_img (np.ndarray): The previous image.
        curr_img (np.ndarray): The current image.
        flow_calculator (callable, optional): A function to compute optical flow.

    Returns:
        np.ndarray: The computed optical flow.
    """
    if flow_calculator is None:
        flow_calculator = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    flow = flow_calculator.calc(prev_img, curr_img, None)
    return flow

def extract_region_magnitudes(flow, region='Body'):
    """
    지정된 영역에서 optical flow의 max와 mean magnitude를 반환
    
    Args:
        flow (np.ndarray): (H, W, 2) optical flow
        region (str): 'Head', 'Body', 'Outer'
        
    Returns:
        tuple: (v, s) = (max_magnitude, mean_magnitude)
    """
    h, w = flow.shape[:2]

    if region == 'Head':
        roi = np.s_[:h//3, :]
    elif region == 'Body':
        roi = np.s_[h//3:2*h//3, :]
    elif region == 'Outer':
        roi = np.s_[2*h//3:, :]
    else:
        roi = np.s_[:, :]  # 전체 영역

    # flow magnitude
    mag = np.linalg.norm(flow, axis=2)  # sqrt(fx² + fy²)
    v = np.max(mag[roi])
    s = np.mean(mag[roi])

    return v, s

def main():
    args = parser.parse_args()
    video_path = args.video_path
    target_fps = args.target_fps
    out_dir = args.out_dir
    
    try:
        folders = os.listdir(video_path)
    except Exception as e:
        print(f"Error reading video directory: {e}")
        sys.exit(1)
    file_paths = glob(os.path.join(video_path, 'A*', '*.mp4'))

    for video_file in tqdm(file_paths, desc="Total Progress"):
        cap = cv2.VideoCapture(video_file)
        if not cap.isOpened():
            print(f"Warning: Could not open video file {video_file}. Skipping.")
            continue
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(round(fps / target_fps))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total_steps = int(frame_count // frame_interval) - 1

        ret, prev = cap.read()
        if not ret:
            print(f"Warning: Could not read first frame from {video_file}. Skipping.")
            cap.release()
            continue
        prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
        pts_src = [[197, 63], [507, 63], [611, 426], [102, 420]]
        prev = homography_transform(pts_src, args.out_size, prev)

        dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        region_seqs = {'Head': {'v': [], 's': []}, 'Body': {'v': [], 's': []}, 'Outer': {'v': [], 's': []}}
        video_name = os.path.basename(os.path.dirname(video_file))
        output_filename = os.path.join(out_dir, f'{video_name}_motion_features.npy')
        if os.path.exists(output_filename):
            print(f"Skipping {video_file}, output already exists.")
            cap.release()
            continue

        for step in tqdm(range(total_steps), desc=f"Frames in {video_name}", leave=False):
            for _ in range(frame_interval - 1):
                cap.read()
            ret, curr = cap.read()
            if not ret:
                break
            curr = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
            curr = homography_transform(pts_src, args.out_size, curr)
            flow = compute_optical_flow(prev, curr, dis)

            region_vs = extract_region_magnitudes_from_box(flow)
            for region in ['Head', 'Body', 'Outer']:
                v, s = region_vs[region]
                region_seqs[region]['v'].append(v)
                region_seqs[region]['s'].append(s)

            prev = curr

        cap.release()
        all_features = {}
        for region in ['Head', 'Body', 'Outer']:
            v_seq = np.array(region_seqs[region]['v'])
            s_seq = np.array(region_seqs[region]['s'])
            features = compute_motion_features_fast(v_seq, s_seq)
            for k, v in features.items():
                all_features[f'{k}_{region}'] = v

        os.makedirs(out_dir, exist_ok=True)
        np.save(output_filename, all_features)

if __name__ == "__main__":
    main()
    sys.exit(0)