import os
import torch
import logging
import matplotlib.pyplot as plt
import numpy as np
logger = logging.getLogger('SleepVST')

def setup_device(device: str, gpu_ids: str = '0') -> torch.device:
    """Sets up the device for PyTorch."""
    if device == 'cuda' and not torch.cuda.is_available():
        logger.error("CUDA is not available. Please check your setup.")
        raise ValueError("CUDA is not available. Please check your setup.")
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids if device != 'cpu' else ''
    return torch.device(device)

def vis_fdiff(orig, diff):
    
    plt.subplot(121),plt.imshow(orig, cmap = 'gray')
    plt.title('Frame'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(diff, cmap = 'gray')
    plt.title('Movement'), plt.xticks([]), plt.yticks([])
    plt.show()