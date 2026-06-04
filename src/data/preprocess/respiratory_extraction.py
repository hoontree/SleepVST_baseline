from pathlib import Path
import numpy as np
from .motion_mag.phase_based import *
import cv2
import matplotlib.pyplot as plt
import copy

from scipy.signal import butter, sosfilt, iirnotch, lfilter
from sklearn.preprocessing import scale

def butter_bandpass(lowcut, highcut, fs, order=1):
    low = lowcut
    high = highcut
    sos  = butter(order, [low, high], btype='band', fs=fs, output='sos', analog=False)
    return sos 

def butter_bandpass_filter(data, lowcut, highcut, fs, order=1):
    sos  = butter_bandpass(lowcut, highcut, fs, order=order)
    y = sosfilt(sos, data)
    return y

def notch_filter(data, f0, Q, fs):
    b, a = iirnotch(f0, Q, fs)
    y = lfilter(b, a, data)
    return y

def filtered_signal(signal, sample_freq, low=0.1, high=0.5, order=1):
    
#     # Notch filter to cancel out the power line disturbance (50 Hz - 60Hz)
#     f0, Q = 50, 5
#     y1 = notch_filter(signal, f0, Q, fs=sample_freq)
#     f0 = 60
#     y2 = notch_filter(y1, f0, Q, fs=sample_freq)
    
    #y2 = signal
    # Butterworth filter for valid freq signals
    #filtered_signal = butter_bandpass_filter(y2, low, high, fs=sample_freq, order=order)
    filtered_signal = butter_bandpass_filter(signal, low, high, fs=sample_freq, order=order)
    
    return filtered_signal

def normalize(y_val):
    if np.max(y_val) - np.min(y_val) == 0:
        return y_val
    y = (y_val - np.min(y_val)) / (np.max(y_val) - np.min(y_val))
    return y


def processing(signal):
    standardize = scale(signal)
    butter = filtered_signal(standardize, 5)
    norm = normalize(butter)
    #norm = normalize(standardize)
    
    return norm

def tracking_movement(video, max_point, bbx_size):
    
    p2, p1 = max_point
    roi = video.copy()
    
    bbx_size = bbx_size // 2
    # print(bbx_size)
    hl, hh, wl, wh = p1-bbx_size, p1+bbx_size, p2-bbx_size, p2+bbx_size
    # bound to zero
    hl = max(0, hl)
    wl = max(0, wl)
    # bound to max
    hh = min(video.shape[2], hh)
    wh = min(video.shape[1], wh)
    # print(hl, hh, wl, wh)
    
    roi = (roi[:, wl:wh, hl:hh]*255).astype(np.uint8)
    # print(roi.shape)
    new_pt = np.array([[[bbx_size, bbx_size]]]).astype(np.float32)
    
    color=np.array([0,255,0])

    
    # Take first frame 
    first_frame = roi[0]
    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_RGB2GRAY)

    prev = new_pt

    # Create a mask image for drawing purposes
    mask = np.zeros_like(first_frame)

    #x_val = [prev[0][0][0]]
    y_val = [prev[0][0][1]]
    for j in range(1, len(roi)):
        # calculate optical flow
        frame = roi[j]
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        #nxt, status, error = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev, None, **lk_params)
        nxt, status, error = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev, None)

        good_new = nxt[0]
        good_old = prev[0]
    
        # draw the tracks
        for i,(new,old) in enumerate(zip(good_new,good_old)):
            a,b = new.ravel()
            c,d = old.ravel()
            mask = cv2.line(mask, (int(a),int(b)),(int(c),int(d)), color.tolist(), 2)
            frame = cv2.circle(frame,(int(a),int(b)),2,color.tolist(),-1)
            #x_val.append(a)
            y_val.append(b)
            
        img = cv2.add(frame, mask)

        # concatenate frames 
        if j == 1:
            of_video = img[np.newaxis,...]
        else:
            of_video = np.concatenate((of_video, img[np.newaxis]))
        
        # Now update the previous frame and previous points
        prev_gray = gray.copy()
        prev = good_new.reshape(-1,1,2)
        
    return of_video, y_val

def make_half(video):
    
    v_len = int(len(video)/2)
    v_new = []
    for i in range(v_len):
        v_new.append(video[i*2])
    v_new = np.array(v_new)
    
    return v_new

def resp_extraction(video, fps, mag_factor, freq_range, attenuate, sigma, temporal_filter, save_dir, epoch_idx,
                    preprocessed_signal_dir=None, record_id=None):

    save_dir = Path(save_dir)

    # Phase based magnification
    no_mag, mag = motionMag(video, mag_factor, freq_range, attenuate, sigma, temporal_filter)

    # get the maximum movement point
    diff = frame_difference(no_mag, mag)
    max_point = get_max_point(diff)

    # Post-processing of magnification
    mag = mag.clip(min=0, max=1)

    # get movement siganl
    mov, mov_normalized = get_movement_signal(video, max_point)
    mag_mov, mag_mov_normalized = get_movement_signal(mag, max_point)

    # save movement signal
    np.save(save_dir / f'epoch_{epoch_idx}_movement.npy', mov)
    np.save(save_dir / f'epoch_{epoch_idx}_magnified_movement.npy', mag_mov)

    # visualization

    # frames
    frame_save_path = save_dir / 'frames'
    frames_with_max_pt(video, max_point, frame_save_path)

    mag_frame_save_path = save_dir / 'mag_frames'
    frames_with_max_pt(mag, max_point, mag_frame_save_path)

    # signals
    num_frames = len(video)

    signal_save_path = save_dir / 'signals'
    signal_with_pt(mov_normalized, num_frames, fps, signal_save_path, epoch_idx,
                   preprocessed_signal_dir, record_id)

    mag_signal_save_path = save_dir / 'mag_signals'
    signal_with_pt(mag_mov_normalized, num_frames, fps, mag_signal_save_path, epoch_idx,
                   preprocessed_signal_dir, record_id)

    return max_point

def resp_extraction_r(video, fps, mag_factor, freq_range, attenuate, sigma, temporal_filter, save_dir, epoch_idx,
                      preprocessed_signal=None, record_id=None):

    save_dir = Path(save_dir)

    # Phase based magnification
    no_mag, mag = motionMag(video, mag_factor, freq_range, attenuate, sigma, temporal_filter)

    # get the maximum movement point
    diff = frame_difference(no_mag, mag)
    max_point = get_max_point(diff)

    # Post-processing of magnification
    mag = mag.clip(min=0, max=1)

    v_new = make_half(video)
    of_video, y_val = tracking_movement(v_new, max_point, bbx_size=50)
    mov = processing(y_val)

    # save movement signal
    np.save(save_dir / f'epoch_{epoch_idx}_movement.npy', mov)

    # visualization

    # # frames
    # frame_save_path = save_dir / 'frames'
    # frames_with_max_pt(video, max_point, frame_save_path)

    # mag_frame_save_path = save_dir / 'mag_frames'
    # frames_with_max_pt(mag, max_point, mag_frame_save_path)

    # signals
    num_frames = len(v_new)

    signal_with_pt(mov, num_frames, fps, save_dir, preprocessed_signal=preprocessed_signal, epoch_idx=epoch_idx-1, record_id=record_id)

    return max_point

def frames_with_max_pt(frames, max_point, save_dir):

    save_dir = Path(save_dir)

    for i in range(len(frames)):
        frame = frames[i]
        pt_on_frame = draw_max_pt(frame, max_point)

        # Convert RGB to BGR for save
        pt_on_frame = cv2.cvtColor(pt_on_frame, cv2.COLOR_RGB2BGR)
        
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f'w_pt_frame_{i}.jpg'
        
        cv2.imwrite(str(save_path), pt_on_frame)

def signal_with_pt(movement, num_frames, fps, save_dir, preprocessed_signal=None, epoch_idx=None, record_id=None):
    """
    Visualize and save movement signal for the entire epoch.
    Optionally compare with preprocessed signal from preprocess_respiratory_edf.py

    - movement: movement signal (1D array)
    - num_frames: number of frames
    - fps: frames per second
    - save_dir: directory to save the signal images
    - epoch_idx: epoch index (optional, required for preprocessed signal comparison)
    - preprocessed_signal_dir: directory containing preprocessed _bw.npy files (optional)
    - record_id: record ID to match preprocessed signal file (optional)
    """
    save_dir = Path(save_dir)

    x = list(range(num_frames))
    time = np.array(x) * 2 / fps

    y_max = np.max(movement) + 0.2*np.max(movement)
    y_min = np.min(movement) - 0.2*np.abs(np.min(movement))

    save_dir.mkdir(parents=True, exist_ok=True)

    # Load preprocessed signal if available
    preprocessed_time = np.arange(len(preprocessed_signal)) / fps if preprocessed_signal is not None else None

    # Save one plot per epoch
    save_path = save_dir / f'signal_epoch_{epoch_idx}.png'

    plt.figure(figsize=(10, 5))
    
    # Plot original movement signal
    plt.plot(time, movement, label='Video-extracted Signal', color='blue', linewidth=1)
    
    # Plot preprocessed signal if available
    if preprocessed_signal is not None:
        plt.plot(preprocessed_time, preprocessed_signal, label='Raw EDF Signal', color='orange', linewidth=1)
        plt.legend(loc='upper right', fontsize=10)
    
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Signal', fontsize=12)
    plt.title(f'Record {record_id} - Epoch {epoch_idx} - Respiratory Signals', fontsize=14)
    plt.grid(True, alpha=0.3)

    plt.savefig(str(save_path), bbox_inches='tight', dpi=200)
    plt.close()


def vis_fdiff(orig, diff):
    
    plt.subplot(121),plt.imshow(orig, cmap = 'gray')
    plt.title('Frame'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(diff, cmap = 'gray')
    plt.title('Movement'), plt.xticks([]), plt.yticks([])
    plt.show()

def draw_max_pt(frame, max_point):
    
    center_coordinates = max_point[1], max_point[0]
    pt_on_frame = copy.deepcopy(frame)
    pt_on_frame = (255 * pt_on_frame).astype(np.uint8)
    pt_on_frame = cv2.circle(pt_on_frame, center_coordinates, radius=3, color=(255,0,0), thickness=-1)
    
    return pt_on_frame

def vis_movement(movement, num_frames, fps):
    
    x = list(range(num_frames))
    time = np.array(x) / fps
    
    y_max = np.max(movement) + 0.2*np.max(movement)
    y_min = np.min(movement) - 0.2*np.abs(np.min(movement))

    plt.plot(time, movement)
    plt.xlabel('time (s)', fontsize=15)
    plt.ylim([y_min, y_max])
    plt.show()
    
def concatenate(num_frames, frame_dir, signal_dir):
    
    concat = []

    for i in range(num_frames):
        frame_path = Path(frame_dir) / 'w_pt_frame_{}.jpg'.format(i)
        frame_img = cv2.imread(frame_path)
        frame_img = cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB)
        frame_img = cv2.resize(frame_img, (300,300))

        signal_path = Path(signal_dir) / 'signal_{}.jpg'.format(i)
        signal_img = cv2.imread(signal_path)
        signal_img = cv2.cvtColor(signal_img, cv2.COLOR_BGR2RGB)
        signal_img = cv2.resize(signal_img, (600,300))
    
        img_concat = cv2.hconcat([frame_img, signal_img])
        concat.append(img_concat)
        
    return concat