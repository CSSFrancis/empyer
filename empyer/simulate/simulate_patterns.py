import numpy as np
from scipy.interpolate import RectBivariateSpline
from empyer.misc.image import rotate
from empyer.misc.kernels import shape_function,sg,random_rotation


def cartesian_to_ellipse(center, angle, lengths):
    """Converts a grid of points to equivalent elliptical points for a spline interpolation

    Parameters
    ---------------
    center: list
        The center of the ellipse
    angle: float
        The angle of the major axis in radians
    lengths: The lengths of the major and minor axis of the ellipse

    Returns
    ------------
    xInd: array-like
        The x indices for the interpolation
    yInd: array-like
        The y indices for the interpolation
    """
    xInd, yInd = np.mgrid[:512, :512]
    major = max(lengths)/np.mean(lengths)
    minor = min(lengths)/np.mean(lengths)
    xInd, yInd = xInd - center[0], yInd - center[1]
    xInd, yInd = rotate(xInd, yInd, angle=-angle)
    xInd, yInd = xInd*minor, yInd*major
    xInd, yInd = rotate(xInd, yInd, angle=angle)
    return xInd, yInd


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


def simulate_symmetry(symmetry=4, I=1, k=4, r=1, iterations=1000):
    """Simulates the intensities of some symmetry for a random rotation.

    Assumes a spherical cluster of randius r and spots appearing at k and an Intensity of I (if the deviation parameter
    "s" is zero)

    Parameters
    ---------------
    symmetry: int
        The symmetry of the clusters
    I: float
        Intensity with deviation parameter of zero
    k: float
        The inverse spacing of the clusters nm^-1
    r: float
        The radius of the cluster.

    Returns
    ------------
    observed_int:list
        The intensities of the speckles described.
    """
    angle = (2*np.pi)/symmetry
    k = [[np.cos(angle*i)*k, np.sin(angle*i)*k, 0] for i in range(symmetry)]
    observed_int = np. zeros(shape=(iterations, symmetry*4))
    for i in range(iterations):
        rotation_vector, theta = random_rotation()
        for j, speckle in enumerate(k):
            s = sg(acc_voltage=200, rotation_vector=rotation_vector, theta=theta, k0=speckle)
            observed_int[i, j*4] = I*shape_function(r=r, s=s)
            observed_int[i, j * 4 + 1] = I * shape_function(r=r, s=s)
    return observed_int


def random_pattern(symmetry, k):
    """Creates a random pattern of some symmetry at some k

    Parameters
    ---------------
    symmetry: int
        The symmetry of the cluster
    k: float
        The inverse spacing of the cluster nm^-1

    Returns
    ------------
    k:array-like
        The
    """
    angle = (2*np.pi)/symmetry
    k = k # +np.random.randn()/10 # normal distribution about k
    k = [[np.cos(angle*i)*k, np.sin(angle*i)*k] for i in range(symmetry)]
    rotation_vector, theta = random_rotation()
    rand_angle = np.random.rand()*np.pi*2
    k = [list(rotate(x, y, rand_angle))+[0] for x, y in k]
    s = [sg(acc_voltage=200, rotation_vector=rotation_vector, theta=theta, k0=speckle) for speckle in k]
    observed_intensity = [100 * shape_function(r=1, s=dev) for dev in s]
    return k, observed_intensity


def simulate_pattern(symmetry, k, num_clusters, probe_size, center, angle, lengths):
    image = np.ones(shape=(512, 512))
    xInd, yInd = np.mgrid[:512, :512]
    for i in range(num_clusters):
        k_val, observed_int = random_pattern(symmetry=symmetry, k=k)
        for pos, inten in zip(k_val, observed_int):
            circle = (xInd - pos[0] - center[0]) ** 2 + (yInd - pos[1] - center[1]) ** 2
            image[circle < probe_size] += inten
            #image[circle < probe_size] += 10

    image = distort(image, center, angle, lengths)
    return image

