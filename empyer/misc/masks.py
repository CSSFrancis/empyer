import numpy as np
from skimage.morphology import watershed, binary_closing, binary_opening, binary_dilation
from skimage.filters import gaussian
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


class Mask(object):
    def __init__(self, obj, is_navigation):
        self.obj = obj
        self.is_navigation = is_navigation

    def __setitem__(self, key, value):
        slices = self.obj._get_array_slices(key, self.is_navigation)
        if self.is_navigation:
            self.obj.metadata.Mask.nav_mask[slices[:-2]] = value
        else:
            self.obj.metadata.Mask.sig_mask[slices[-2:]] = value

    def __getitem__(self, key, out=None):
        if self.is_navigation:
            return self.obj.metadata.Mask.nav_mask[key]
        else:
            return self.obj.metadata.Mask.sig_mask[key]

    def __str__ (self):
        if self.is_navigation:
            return str(self.obj.metadata.Mask.nav_mask)
        else:
            return str(self.obj.metadata.Mask.sig_mask)

    def mask_circle(self, center, radius, unmask=False):
        """Applies a mask to every pixel using a shape and the appropriate definition

        Parameters
        ----------
        center: tuple
            The center of the circle to be masked
        radius: float or int
            The radius of the circle. Ints result in index like radius, floats use the scale defined in the axes manager
        unmask: bool
            Unmask the pixels defined by the mask.

        Retunrns
        ---------
        mask: ndarray
            The mask added to the image
        """
        if not all(isinstance(item, int) for item in center):
            center = (self.obj.axes_manager.signal_axes[1].value2index(center[1]),
                      self.obj.axes_manager.signal_axes[0].value2index(center[0]))
        if not isinstance(radius, int):
            radius = self.obj.axes_manager.signal_axes[0].value2index(radius)
        if self.is_navigation:
            dimensions = self.obj.axes_manager.navigation_shape
        else:
            dimensions = self.obj.axes_manager.signal_shape
        mask = _circular_mask(center, radius, dimensions)

        if self.is_navigation:
            if unmask:
                self.obj.metadata.Mask.nav_mask = self.obj.metadata.Mask.nav_mask ^ mask
            else:
                self.obj.metadata.Mask.nav_mask = self.obj.metadata.Mask.nav_mask + mask
        else:
            if unmask:
                self.obj.metadata.Mask.sig_mask = self.obj.metadata.Mask.sig_mask ^ mask
            else:
                self.obj.metadata.Mask.sig_mask = self.obj.metadata.Mask.sig_mask + mask

        return mask

    def mask_reset(self):
        """Resets the mask to completely unmasked.
        """
        if self.is_navigation:
            self.obj.metadata.Mask.nav_mask = np.zeros(shape=self.obj.axes_manager.navigation_shape, dtype=bool)
        else:
            self.obj.metadata.Mask.sig_mask = np.zeros(shape=self.obj.axes_manager.signal_shape, dtype=bool)
        return

    def mask_rings(self, center, inner_radius,outer_radius, unmask=False):
        """Applies a mask a ring shape

            Parameters
            ----------
            center: tuple
                The center of the circle to be masked
            inner_radius: (float or int)
                The inner radius of the circle. Ints result in index like radius, floats use the scale defined in the
                axes manager
            outer_radius: (float or int)
                The outer radius of the circle. Ints result in index like radius, floats use the scale defined in the
                axes manager
            unmask: bool
                Unmask the pixels defined by the mask.

            Returns
            ---------
            mask: ndarray
                The mask added to the image
        """
        if not all(isinstance(item, int) for item in center):
            center = (self.obj.axes_manager.signal_axes[1].value2index(center[1]),
                      self.obj.axes_manager.signal_axes[0].value2index(center[0]))
        if not isinstance(outer_radius, int):
            outer_radius = self.obj.axes_manager.signal_axes[0].value2index(outer_radius)
        if not isinstance(inner_radius, int):
            inner_radius = self.obj.axes_manager.signal_axes[0].value2index(inner_radius)

        if self.is_navigation:
            dimensions = self.obj.axes_manager.navigation_shape
            mask = _ring_mask(center=center, inner_radius=inner_radius, outer_radius=outer_radius, dim=dimensions)
            if unmask:
                self.obj.metadata.Mask.nav_mask = self.obj.metadata.Mask.nav_mask ^ mask
            else:
                self.obj.metadata.Mask.nav_mask = self.obj.metadata.Mask.nav_mask + mask
        else:
            dimensions = self.obj.axes_manager.signal_shape
            mask = _ring_mask(center=center, inner_radius=inner_radius, outer_radius=outer_radius, dim=dimensions)
            if unmask:
                self.obj.metadata.Mask.sig_mask = self.obj.metadata.Mask.sig_mask ^ mask
            else:
                self.obj.metadata.Mask.sig_mask = self.obj.metadata.Mask.sig_mask + mask
        return mask

    def mask_beam_stop(self, method="hist", mask=False, **kwargs):
        """Creates and adds a mask for the beam stop. Based on the summed image.
        Parameters:
        ------------
        method: str
            Only 'hist' at the moment
        mask: bool
            Set the values in the mask equal to this value
        axis: str
            Change the signal axis
        """
        if self.is_navigation:
            print("Beam stop masking only works on signal axes.")
            return
        else:
            mask = _beam_stop_mask(self.obj.metadata.Sum.sig_sum, method=method, **kwargs)
        return mask

    def mask_border(self, pixels=1, unmask=False):
        if not isinstance(pixels, int):
            pixels = (self.axes_manager.signal_axes[0].value2index(pixels))
        self[..., -pixels:] = not unmask
        self[..., : pixels] = not unmask
        self[..., : pixels, :] = not unmask
        self[..., -pixels:, :] = not unmask






