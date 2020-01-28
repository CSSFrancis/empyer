import numpy as np
from skimage.morphology import watershed, binary_closing, binary_opening, binary_dilation
from skimage.filters import gaussian
from skimage.draw import circle
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.signal import savgol_filter


def _circular_mask(center, radius, dim):
    x, y = np.ogrid[-center[0]:dim[0]-center[0], -center[1]:dim[1]-center[1]]
    mask = x * x + y * y <= radius * radius
    return mask


def _rectangular_mask(x1, x2, y1, y2, dim):
    """Returns a mask with masked values based on x1,x2, y1, and y.

    Parameters
    ------------
    x1: int
    x2: int
    y1 : int
    y2: int

    Returns
    ----------
    mask: ndarray
        The resulting boolean mask for the rectangle described.
    """
    mask = np.zeros(dim, dtype=bool)
    mask[x1:x2, y1:y2]=True
    return mask


def _ring_mask(center, outer_radius, inner_radius, dim):
    x, y = np.ogrid[-center[0]:dim[0]-center[0], -center[1]:dim[1]-center[1]]
    outer_mask = x * x + y * y <= outer_radius * outer_radius
    inner_mask = x * x + y * y <= inner_radius * inner_radius
    mask = outer_mask ^ inner_mask
    return mask


def _beam_stop_mask(summed_image, method="watershed", **kwargs):
    """These are some methods I have developed for blocking the beam stop with a mask.  Use them with some caution.
    It is very possible they don't work on every case and the parameters may have to be adjusted for every system.
    :param summed_image:
    :param method:
    :param kwargs:
    :return:

    Notes
    ----------
    This function has the methods:
    "watershed": Uses the watershed algorithum to find a continious region with minimum values and assumes that region
    is the beam stop

    "hist": Uses a histagram of values and uses the most well defined local minimum.  Then closes that region to make a
    continous region.
    """
    if method is "watershed":
        smoothed_image = gaussian(summed_image, sigma=4)
        plt.imshow(smoothed_image)
        plt.show()
        w = watershed(smoothed_image, 5, **kwargs)
        plt.imshow(w)
        plt.show()
    if method is "hist":
        hist = np.histogram(np.reshape(summed_image, -1), 1000)
        inv_hist = abs(hist[0] - max(hist[0]))
        inv_hist = savgol_filter(inv_hist, 15, 3)
        p = max(hist[0])/3
        minvalues = find_peaks(inv_hist, prominence=p)
        cutoff = hist[1][minvalues[0][0]]
        print("The cutoff should be: ", hist[1][minvalues[0][0]])
        mask = summed_image < cutoff
        selem = np.array([[0, 1, 1, 1, 0], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [0, 1, 1, 1, 0]],
                        dtype=bool)
        mask = binary_closing(mask, selem=selem)
        mask = binary_opening(mask, selem=selem)
        mask = binary_dilation(mask, selem=selem)
        return mask


def _determine_center(summed_image, mask):
    return





