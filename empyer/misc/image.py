import numpy as np


def bin_2d(image, binning_factor):
    """
    -This function takes a 2d image and then linearly interpolates to find the intermediate value.
    :param image:2d numpy array
    :param binning_factor: factor for binning down image
    :return: new_image: 2d image binned down by binning factor
    """
    sh = np.shape(image)

    dim1_cut = sh[0]%binning_factor
    dim2_cut = sh[1]%binning_factor
    cut_image = image[dim1_cut:, dim2_cut:]  # in case the image is not a multiple of the binning factor

    new_image = cut_image.reshape(sh[0] // binning_factor, binning_factor, sh[1] // binning_factor, binning_factor).mean(
        -1).mean(1)

    return new_image


def cartesian_to_polar(x, y, center):
    """
    A function that converts the x,y coordinates to polar ones.
    -Does not do the circular correction
    """
    corrected_x = x-center[0]
    corrected_y = y-center[1]
    theta = np.arctan2(corrected_y, corrected_x)
    r = np.sqrt(corrected_x**2 + corrected_y**2)
    return theta, r


def polar_to_cartesian(r, theta, center):
    """
    A function that converts polar (r,theta) coordinates to cartesian(x,y) ones
    """
    x = center[0] + r*np.cos(theta)
    y = center[1] + r*np.sin(theta)
    return x, y


def cartesian_list_to_polar(x_list,y_list,center):
    """
    A function that converts a list of x,y coordinates to (r,theta)
    """
    theta_list = []
    r_list = []
    for x, y in zip(x_list,y_list):
        t, r = cartesian_to_polar(x,y,center)
        theta_list.append(t)
        r_list.append(r)
    return theta_list, r_list


def polar_list_to_cartesian(r_list, theta_list, center):
    x_list = []
    y_list = []
    for r, t in zip(r_list, theta_list):
        x, y = polar_to_cartesian(r, t, center)
        x_list.append(x)
        y_list.append(y)
    return x_list, y_list


def create_grid(dimension1, dimension2):
    """
    Takes data from array of points turns it into a grid of coordinates
    """
    dim1, dim2 = np.meshgrid(dimension1, dimension2)
    size = len(dimension1)*len(dimension2)
    a, b = np.reshape(dim1, size), np.reshape(dim2,size)
    return a, b


def ellipsoid_list_to_cartesian(r_list, theta_list, center, major, minor, angle, even_spaced=False):
    """
    :param r_list: list of all of the radius.  Can either be all values or even_spaced
    :param theta_list: list of all of the theta.  Can either be all values or even_spaced
    :param center:
    :param major:
    :param minor:
    :param angle:
    :param even_spaced: if the r and t values are evenly spaced.  Speeds computation a lot...
    :return: list of equivalent x and y values.
    """
    x_list = []
    y_list = []
    focii_ratio = minor / major
    h_o = (1 / ((focii_ratio ** 2) + 1) ** 0.5)  # finding the major and minor axis lengths
    k_o = (1 / ((focii_ratio ** -2) + 1) ** 0.5)
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    if even_spaced:
        t_sin = [np.sin(t)for t in theta_list]
        t_cos = [np.cos(t)for t in theta_list]
        h_list = np.multiply(r_list, h_o)
        k_list = np.multiply(r_list, k_o)
        x_unrotated = [[h*tc - h*ts for ts, tc in zip(t_sin, t_cos)]for h in h_list]
        y_unrotated = [[k*tc + k*ts for ts, tc in zip(t_sin, t_cos)] for k in k_list]
        x_list = np.subtract(np.multiply(x_unrotated, cos_angle), np.multiply(y_unrotated, sin_angle))
        x_list = np.add(x_list, center[0])
        y_list = np.add(np.multiply(y_unrotated, cos_angle), np.multiply(x_unrotated, sin_angle))
        y_list = np.add(y_list, center[1])
    else:
        for r, t in zip(r_list,theta_list):
            h = h_o *r
            k = k_o *r
            x_unrotated = h*(np.cos(t)) - h*np.sin(t)
            y_unrotated = k*(np.sin(t)) + k*np.cos(t)
            #  need to rotate by the angle
            x = center[0] + x_unrotated * cos_angle - y_unrotated * sin_angle
            y = center[1]+y_unrotated*cos_angle + x_unrotated*sin_angle
            x_list.append(x)
            y_list.append(y)
    return x_list, y_list
