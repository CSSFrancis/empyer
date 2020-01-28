import numpy as np
from skimage.morphology import watershed
from skimage.filters import s

def _circular_mask(center, radius, x_len,y_len, x_start=0, y_start=0, scale=1):
    x, y = np.ogrid[x_start-center[0]:x_len-center[0]:scale, y_start-center[1]:y_len-center[1]:scale]
    mask = x * x + y * y <= radius * radius
    return mask


def _rectangular_mask(x1,x2,y1,y2, x_len,y_len, x_start=0, y_start=0, scale=1):
    x, y = np.ogrid[x_start:x_len:scale, y_start:y_len:scale]
    x_mask = x1<=x<=x2
    y_mask = y1<=y<=y2
    mask = x_mask+y_mask
    return mask


def _ring_mask(center, outer_radius, inner_radius, x_len,y_len, x_start=0, y_start=0, scale=1):
    x, y = np.ogrid[x_start-center[0]:x_len-center[0]:scale, y_start-center[1]:y_len-center[1]:scale]
    outer_mask = x * x + y * y <= outer_radius * outer_radius
    inner_mask = x * x + y * y <= inner_radius * inner_radius
    mask = outer_mask-inner_mask
    return mask


def _beam_stop_mask(summed_image, method="watershed", **kwargs):
    if method is "watershed":
        watershed(summed_image, 1, **kwargs)
    if method is "min":

def _determine_center(summed_image, mask,):





