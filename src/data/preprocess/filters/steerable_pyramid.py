"""
Complex Steerable Pyramid Filters for Phase-Based Motion Magnification
"""

import numpy as np
from typing import Tuple, List


def get_polar_grid(dims):
    center = np.ceil((np.array(dims))/2).astype(int)
    xramp, yramp = np.meshgrid(np.linspace(-1, 1, dims[1]+1)[:-1], np.linspace(-1, 1, dims[0]+1)[:-1])
    theta = np.arctan2(yramp, xramp)
    r = np.sqrt(xramp**2 + yramp**2)
    
    # eliminate the zero at the center
    r[center[0], center[1]] = min((r[center[0], center[1]-1], r[center[0]-1, center[1]]))/2
    
    return theta, r


def get_radial_mask_pair(r, rad, t_width):
    log_rad = np.log2(rad)-np.log2(r)
    hi_mask = abs(np.cos(log_rad.clip(min=-t_width, max=0)*np.pi/(2*t_width)))
    lo_mask = np.sqrt(1-(hi_mask**2))
    
    return (hi_mask, lo_mask)


def get_angle_mask(b, orientations, angle):
    order = orientations - 1
    a_constant = np.sqrt((2**(2*order))*(np.math.factorial(order)**2)/(orientations*np.math.factorial(2*order)))
    angle2 = simplify_phase(angle - (np.pi*b/orientations))
    return 2*a_constant*(np.cos(angle2)**order)*(abs(angle2) < np.pi/2)


def max_pyr_height(size: int) -> int:
    """
    Compute maximum pyramid height based on image size.

    Args:
        size: Minimum dimension of the image

    Returns:
        Maximum number of pyramid levels
    """
    return int(np.log2(min(size))) - 2


def get_filters(dims, r_vals=None, orientations=2, only_ver=False, t_width=1):
    """
    Gets a steerbale filter bank (ndarrays)
    - dims: (h, w). Dimensions of the output filters. 
            Should be the same size as the image you're using these to filter
    - r_vals: The boundary between adjacent filters. 
              Should be an array.
              e.g.: 2**np.array(list(range(0,-7,-1)))
    - orientations: The number of filters per level
    - t-width: The falloff of each filter. 
               Smaller t_widths correspond to thicker filters with less falloff
    - only_ver: get filters of vertical direction.
    """
    if r_vals is None:
        r_vals = 2**np.array(list(range(0,-max_pyr_height(dims)-1,-1)), dtype=float)
        
    angle, r = get_polar_grid(dims)
    hi_mask, lo_mask_prev = get_radial_mask_pair(r_vals[0], r, t_width)
    
    filters = [hi_mask]
    for i in range(1, len(r_vals)):
        hi_mask, lo_mask = get_radial_mask_pair(r_vals[i], r, t_width)
        rad_mask = hi_mask * lo_mask_prev
        
        for j in range(orientations):
            angle_mask = get_angle_mask(j, orientations, angle)
            angle_mask = np.rot90(angle_mask, 2) # add for rotation of filter
            filters += [rad_mask*angle_mask/2]
        lo_mask_prev = lo_mask
        
    filters += [lo_mask_prev]
    
    if only_ver == True:
        fil_idx = [i for i in range(len(filters)) if (i % 2 == 0)]
        fil_idx.append(len(filters)-1)
        filters = [filters[idx] for idx in fil_idx]
    
    return filters


def simplify_phase(x):
    '''
    Moves x into the [-pi, pi] range.
    '''
    phase = ((x + np.pi) % (2*np.pi)) - np.pi
    return phase
