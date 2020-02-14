import numpy as np
from scipy.interpolate import RectBivariateSpline
import time

from empyer.misc.image import ellipsoid_list_to_cartesian


def convert(img, mask, center=None, angle=None, lengths=None, radius=[0,100], phase_width=720, normalized=False):
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
    initial_y, initial_x = range(0, img_shape[-2]), range(0, img_shape[-1])
    if center is None:
        center = np.true_divide(img_shape[-2:], 2)
    final_the = np.linspace(0, 2*np.pi, num=phase_width)
    final_rad = np.arange(radius[0], radius[1], 1)
    final_x, final_y = ellipsoid_list_to_cartesian(final_rad,
                                                   final_the,
                                                   center,
                                                   axes_lengths=lengths,
                                                   angle=angle)
    intensity = img
    # setting masked values to negative values. Anything interpolated from masked values becomes negative
    if mask is not None:
        intensity[mask] = -999999
    tic =time.time()
    spline = RectBivariateSpline(initial_x, initial_y, intensity, kx=1, ky=1)  # bi-linear spline (Takes 90% of time)
    toc =time.time()
    print("time is:", toc-tic)
    tic = time.time()
    polar_img = np.array(spline.ev(final_x, final_y))
    toc = time.time()
    print("time is:", toc - tic)
    polar_img = np.reshape(polar_img, (int(radius[1]-radius[0]), phase_width))
    polar_img[polar_img < -10] = mask
    print(spline.get_coeffs()[0])
    print(spline.integral(1, 2, 1, 2))
    if normalized:
        ellipse_grid = np.reshape(list(zip(np.reshape(final_x[:,0:2], -1), np.reshape(final_y[:,0:2], -1))),
                                  (int(radius[1]-radius[0]),2,2)) # add zero column and then do fancy vector stuff takin
        #the vectors which cross the paralellegram.
        [ellipse_grid[i][0]for i in range(int(radius[1]-radius[0])-1)]



    return polar_img, mask



