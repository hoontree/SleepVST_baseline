from ..filters.steerable_pyramid import get_filters, max_pyr_height, simplify_phase
from ..filters.temporal_filters import amplitude_weighted_blur
import numpy as np
import scipy.fft as sfft
from tqdm import tqdm

yiq_from_rgb = (np.array([[0.29900000,  0.58700000,  0.11400000],
                          [0.59590059, -0.27455667, -0.32134392],
                          [0.21153661, -0.52273617,  0.31119955]])).astype(np.float32)
rgb_from_yiq = np.linalg.inv(yiq_from_rgb)


def rgb2yiq(rgb_image):
    image = rgb_image.astype(np.float32)
    #return image.dot(yiq_from_rgb.T) 
    return image @ yiq_from_rgb.T

def yiq2rgb(yiq_image):
    image = yiq_image.astype(np.float32)
    #return image.dot(rgb_from_yiq.T)
    return image @ rgb_from_yiq.T

def motionMag(video, mag_factor, freq_range, attenuate, sigma, temporal_filter):
    
    # Video Info
    num_frames, w, h, num_ch = video.shape
    
    # Get YIQ video
    yiq_video = rgb2yiq(video)
    
    # Get FFT video
    fft_video = np.zeros((num_frames, w, h), dtype=np.complex64)
    for i in range(num_frames):
        fft_video[i] = sfft.fftshift(sfft.fft2(yiq_video[i][:,:,0]))
    
    # Pyramid height
    pyr_height = max_pyr_height((w, h))
    
    # Complex Steerable Filters
    filters = get_filters((w, h), 2**np.array(list(range(0,-pyr_height-1,-1)), dtype=float), 2, True)    
    
    # Magnification
    
    ## Magnified y channel
    mag_y_channel = np.zeros((num_frames, w, h), dtype=np.complex64)
    ## y channel w/o magnification
    y_channel = np.zeros((num_frames, w, h), dtype=np.complex64)
    
    dc_frame_idx = 0
    fl, fh = freq_range
    for i in range(1,len(filters)-1):

        dc_frame = sfft.ifft2(sfft.ifftshift(filters[i]*fft_video[dc_frame_idx]))    
        dc_frame_no_mag = dc_frame / np.abs(dc_frame)    
        dc_frame_phase = np.angle(dc_frame)

        total = np.zeros(fft_video.shape, dtype=float)
        filtered = np.zeros(fft_video.shape, dtype=np.complex64)

        for n in range(num_frames):
            filtered[n] = sfft.ifft2(sfft.ifftshift(filters[i]*fft_video[n]))
            total[n] = simplify_phase(np.angle(filtered[n]) - dc_frame_phase)

        ## Temporal Filter
        total = temporal_filter(total, fl, fh).astype(float)

        for n in range(num_frames):
            phase_of_frame = total[n]
            if sigma != 0:
                phase_of_frame = amplitude_weighted_blur(phase_of_frame, np.abs(filtered[n]), sigma)

            ## Magnified phase
            mag_phase_of_frame = phase_of_frame * mag_factor
            #phase_of_frame *= mag_factor

            if attenuate:
                temp_orig = np.abs(filtered[n])*dc_frame_no_mag
            else:
                temp_orig = filtered[n]
            
            no_mag_component = 2*filters[i]*sfft.fftshift(sfft.fft2(temp_orig*np.exp(1j*phase_of_frame)))
            magnified_component = 2*filters[i]*sfft.fftshift(sfft.fft2(temp_orig*np.exp(1j*mag_phase_of_frame)))

            y_channel[n] = y_channel[n] + no_mag_component
            mag_y_channel[n] = mag_y_channel[n] + magnified_component

            
    for i in range(num_frames):
        y_channel[i] = y_channel[i] + (fft_video[i]*(filters[-1]**2))
        mag_y_channel[i] = mag_y_channel[i] + (fft_video[i]*(filters[-1]**2))

            
    out_mag = np.zeros(yiq_video.shape)
    out_no_mag = np.zeros(yiq_video.shape)
    for i in range(num_frames):
        out_frame  = np.dstack((np.real(sfft.ifft2(sfft.ifftshift(y_channel[i]))), yiq_video[i,:,:,1:3])) 
        out_mag_frame  = np.dstack((np.real(sfft.ifft2(sfft.ifftshift(mag_y_channel[i]))), yiq_video[i,:,:,1:3]))
        
        out_no_mag[i] = out_frame
        out_mag[i] = out_mag_frame
                  
    out_no_mag = yiq2rgb(out_no_mag)
    out_mag = yiq2rgb(out_mag)
    
    return out_no_mag, out_mag
    #return out_no_mag.clip(min=0, max=1), out_mag.clip(min=0, max=1)
    #return out

def frame_difference(no_mag, mag):
    
    # Calculate frame difference
    diff = np.abs(mag - no_mag)
    
    # Difference sum
    diff_sum = diff.sum(axis=0)
    diff_sum = diff_sum.sum(axis=-1)
    
    return diff_sum

def get_max_point(diff):
    
    max_pt = np.where(diff == np.max(diff))
    
    return max_pt[0][0], max_pt[1][0]

def get_movement_signal(video, max_point):
    
    p1, p2 = max_point
    traj = video[:, p1, p2, :]
    movement = traj.sum(axis=-1)
    
    mov_normalized = (movement - np.mean(movement)) / np.std(movement)
    
    return movement, mov_normalized