from skimage.filters import gaussian
import numpy as np

def difference_of_iir(delta, rl, rh):
    lowpass_1 = delta[0].copy()
    lowpass_2 = lowpass_1.copy()
    out = np.zeros(delta.shape, dtype=delta.dtype)
    for i in range(1, delta.shape[0]):
        lowpass_1 = (1-rh)*lowpass_1 + rh*delta[i]
        lowpass_2 = (1-rl)*lowpass_2 + rl*delta[i]
        out[i] = lowpass_1 - lowpass_2
    return out

def amplitude_weighted_blur(x, weight, sigma):
    '''
    Where x is phase of the frame ,weight is total amplitude of the frame, 
    sigma is Standard deviation for Gaussian kernel.
    The mode parameter determines how the array borders are handled.
    '''
    if sigma != 0:
        return gaussian(x*weight, sigma, mode="wrap") / gaussian(weight, sigma, mode="wrap")
    return x