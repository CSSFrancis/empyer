import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import map_coordinates
from empyer.misc.image import ellipsoid_list_to_cartesian
import time


def to_polar_image(img, center=None, angle=None, lengths=None, radius=[0, 100], phase_width=720, normalized=False):
    """ Function for converting an image in cartesian coordinates to polar coordinates.

    Parameters
    ------------------
    img:array-like
        A n by 2-d array for the image to convert to polar coordinates
    center: list
        [X,Y] coordinates for the center of the image
    angle: float
        Angle of rotation if the sample is elliptical
    lengths: list
        The major and minor lengths of the ellipse
    radius: list
        The inner and outer indexes to define the radius by.
    phase_width: int
        The number of "pixels" in the polar image along the x direction
    normalized: bool
        If True the intensity is conserved in the image.  If there is another normalization step this isn't necessarily
        needed.

    Returns
    -----------
    polar_img: array-like
        A numpy array of the input img  in polar coordiates. Dim (radius[1]-radius[0]) x phase_width
    """
    img_shape = np.shape(img)
    if center is None:
        center = np.true_divide(img_shape[-2:], 2)
    final_the = np.linspace(0, 2*np.pi, num=phase_width)
    final_rad = np.arange(radius[0], radius[1], 1)
    final_x, final_y = ellipsoid_list_to_cartesian(final_rad,
                                                   final_the,
                                                   center,
                                                   axes_lengths=lengths,
                                                   angle=angle)

    pol = np.array([final_x,final_y])
    polar_img = map_coordinates(img, pol, order=1)
    if normalized:
        # Normalizing based on pixels. This involves knowing the area belonging to each pixel.
        polar_img = [rad_pix*(rad+rad-1)*np.pi/phase_width for rad_pix,rad in zip(polar_img[:],final_rad) if final_rad is not 0]

    return polar_img



