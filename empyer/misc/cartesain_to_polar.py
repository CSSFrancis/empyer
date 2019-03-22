
import numpy as np
from scipy.interpolate import RectBivariateSpline

from empyer.misc.image import ellipsoid_list_to_cartesian, polar_list_to_cartesian, create_grid


def convert(img, mask=None, center=None, angle=None, foci=None, radius=None, phase_width=720):
    #  This function someday will be faster hopefully...
    """
    :param img: a n by 2-d array for the image to convert to polar coordinates
    :param mask: 2-d boolean array which excludes certain points
    :param center: x,y coordinates for the center of the image
    :param angle: angle of rotation if the sample is elliptical
    :param foci: The lengths of the foci of the ellipse
    :param phase_width: the number of "pixels" in the polar image along the x direction
    :param radius: the number of "pixels" in the polar image along the y direction
    :param plot: Plot the image after converting it...
    :return: polar_img r vs. theta
    """
    img_shape = np.shape(img)
    if center is None:
        center = np.true_divide(img_shape[-2:], 2)
    center = [center[1], center[0]]  # converting to y,x or array coordinates
    if radius is None:
        radius = int(min(img_shape[-2:]) - max(np.abs(np.subtract(img_shape[-2:], center)))-5)
        if foci is not None:
            radius = int(radius/(max(foci)/min(foci)))
    # setting up grids for faster interpolation
    r_inital = 1
    r_final = int(radius + r_inital)

    initial_y, initial_x = range(1, img_shape[-2]+1), range(1, img_shape[-1]+1)
    # for perfectly circular conversions
    if angle is None and foci is None:
        final_theta, final_r = create_grid(np.linspace(-1 * np.pi, np.pi, phase_width), np.arange(r_inital, r_final, 1))
        final_x, final_y = polar_list_to_cartesian(final_r, final_theta, center)
    # for elliptical conversions
    else:
        angle = (np.pi/2)-angle  # converting to y,x or array coordinates
        final_the = np.linspace(-1*np.pi, np.pi, num=phase_width)
        final_rad = np.arange(r_inital, r_final, 1)
        final_x, final_y = ellipsoid_list_to_cartesian(final_rad,
                                                       final_the,
                                                       center,
                                                       major=foci[0],
                                                       minor=foci[1],
                                                       angle=angle,
                                                       even_spaced=True)
    inten = img

    # setting masked values to negative values. Anything interpolated from masked values becomes negative
    if mask is not None:
        inten[mask] = -999999
    # For higher than 2 dimensional arrays.  Speeds up computations a little but requires more memory
    if len(img_shape) > 2:
        inten = np.reshape(inten, (-1, *img_shape[-2:]))
        polar_img = np.zeros((inten.shape[0],)+(r_final - r_inital, phase_width))
        for i, img in enumerate(inten):
            spline = RectBivariateSpline(initial_x, initial_y, img, kx=1, ky=1)  # bi-linear spline
            polar = np.array(spline.ev(final_x, final_y))
            polar_img[i] = np.reshape(polar, [r_final - r_inital, phase_width])
        polar_img = np.reshape(polar_img, (*img_shape[:-2], radius, phase_width))

    else:
        spline = RectBivariateSpline(initial_x, initial_y, inten, kx=1, ky=1)  # bi-linear spline
        polar_img = np.array(spline.ev(final_x, final_y))
        polar_img = np.reshape(polar_img, (radius, phase_width))

    # outputting new mask
    if mask is not None:
        polar_mask = polar_img < 0
        polar_img[polar_img < 0] = 0

    else:
        polar_mask = None
    return polar_img



