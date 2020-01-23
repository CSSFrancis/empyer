import numpy as np


def flatten_axis(array, axis):
    array_shape = np.shape(array)
    np.reshape(array, (*array_shape[:axis-1], -1, *array_shape[-2:]))


def square(array):
    return np.power(array, 2)


def bin_2d(image, binning_factor):
    """Binning a 2-dimensional image  by some factor

    Parameters
    ----------
    image : 2-d array

    binning_factor : int

    Returns
    ----------
    new_image : 2-d array
    """
    sh = np.shape(image)

    dim1_cut = sh[0]%binning_factor
    dim2_cut = sh[1]%binning_factor
    cut_image = image[dim1_cut:, dim2_cut:]  # in case the image is not a multiple of the binning factor

    new_image = cut_image.reshape(sh[0] // binning_factor, binning_factor, sh[1] // binning_factor, binning_factor).mean(
        -1).mean(1)

    return new_image


def cartesian_to_polar(x, y, center):
    """A function that converts the x,y coordinates to polar ones.
    -Does not do the circular correction
    """
    corrected_x = x-center[0]
    corrected_y = y-center[1]
    theta = np.arctan2(corrected_y, corrected_x)
    r = np.sqrt(corrected_x**2 + corrected_y**2)
    return theta, r


def polar_to_cartesian(r, theta, center):
    """A function that converts polar (r,theta) coordinates to cartesian(x,y) ones

    Parameters
    ----------
    r: float
        radius
    theta: float
        angle
    center: array
        center of the array [x,y]
    Returns
    ----------
    x: float
    y: float
    """
    x = center[0] + r*np.cos(theta)
    y = center[1] + r*np.sin(theta)
    return x, y


def cartesian_list_to_polar(x_list,y_list,center):
    """A function that converts a list of x,y coordinates to (r,theta)

    Parameters
    ----------
    r: float
        radius
    theta: float
        angle
    center: array
        center of the array [x,y]
    Returns
    ----------
    theta_list: 2-d array
    r_list: 2-d array
    """
    theta_list = []
    r_list = []
    for x, y in zip(x_list,y_list):
        t, r = cartesian_to_polar(x,y,center)
        theta_list.append(t)
        r_list.append(r)
    return theta_list, r_list


def polar_list_to_cartesian(r_list, theta_list, center):
    """
    Parameters
    ----------
    r_list: array_like
        radius
    theta_list: array_like
        angle
    center: array
        center of the array [x,y]
    Returns
    ----------
    x_list: array
    y_list: array

    """
    x_list = []
    y_list = []
    for r, t in zip(r_list, theta_list):
        x, y = polar_to_cartesian(r, t, center)
        x_list.append(x)
        y_list.append(y)
    return x_list, y_list


def create_grid(dimension1, dimension2):
    """
    Parameters
    ----------
    dimension1 : array
    dimension2 : array

    Returns
    ----------
    a: array
    b: array
    """
    dim1, dim2 = np.meshgrid(dimension1, dimension2)
    size = len(dimension1)*len(dimension2)
    a, b = np.reshape(dim1, size), np.reshape(dim2,size)
    return a, b


def ellipsoid_list_to_cartesian(r_list, theta_list, center, axes_lengths=None, angle=None):
    """Takes a list of ellipsoid points and then use then find their cartesian equivalent

    Parameters
    ----------
    r_list: array
        list of all of the radius.  Can either be all values or even_spaced
    theta_list: array
        list of all of the radius.  Can either be all values or even_spaced
    center: array_like
        center of the ellipsoid
    lengths: float
        length of the major axis
    minor: float
        length of the minor axis
    angle: float
        angle of the major axis in radians

    Returns
    ----------
    x_list: array_like
        list of x points
    y_list: array_like
        list of y points
    """

    # Averaging the major and minor axes
    if axes_lengths is not None:
        axes_avg = sum(axes_lengths)/2
        h_o = max(axes_lengths)/axes_avg  # major
        k_o = min(axes_lengths)/axes_avg
    else:
        h_o = 1
        k_o = 1
    r_mat = np.mat(r_list)

    # calculating points equally spaced annularly on a unit circle
    t_sin = np.mat([np.sin(t)for t in theta_list])
    t_cos = np.mat([np.cos(t)for t in theta_list])
    # unit circle to ellipses at r spacing
    x_circle = r_mat.transpose()*t_sin*h_o
    y_circle = r_mat.transpose()*t_cos * k_o

    if angle is not None:
        # angle of rotation
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        x_list = x_circle*cos_angle - y_circle*sin_angle
        x_list = np.add(x_list, center[0])
        y_list = y_circle*cos_angle + x_circle*sin_angle
        y_list = np.add(y_list, center[1])
        return np.array(x_list), np.array(y_list)
    else:
        x_list = np.add(x_circle, center[0])
        y_list = np.add(y_circle, center[1])
        return np.array(x_list), np.array(y_list)


def random_ellipse(num_points, center, foci, angle):
    rand_angle = np.random.rand(num_points) * 2 * np.pi  # random points on a circle
    points = [[(np.cos(ang) * foci[0]), np.sin(ang) * foci[1]] for ang in rand_angle]  # circle to ellipse
    points = np.array([[round(x * np.cos(angle) - y * np.sin(angle) + center[0]),
                       round(y * np.cos(angle) + x * np.sin(angle) + center[1])]  # rotating the ellipse
                       for x, y in points], dtype=int)

    return points


def rotate(x, y, angle):
    return x*np.cos(angle)-y*np.sin(angle), y*np.cos(angle)+x*np.sin(angle)


def distort(image, center, angle, lengths):
    """Takes an image and distorts the image based on an elliptical distortion

    Parameters
    ---------------
    image: array-like
        The image to apply the elliptical distortion to
    center: list
        The center of the ellipse
    angle: float
        The angle of the major axis in radians
    lengths: The lengths of the major and minor axis of the ellipse

    Returns
    ------------
    distorted:array-like
        The elliptically distorted image
    """
    img_shape = np.shape(image)
    initial_y, initial_x = range(-center[1], img_shape[-2]-center[1]), range(-center[0], img_shape[-1]-center[0])
    spline = RectBivariateSpline(initial_x, initial_y, image, kx=1, ky=1)
    xInd, yInd = cartesian_to_ellipse(center=center, angle=angle, lengths=lengths)
    distorted = np.array(spline.ev(yInd, xInd))
    return distorted