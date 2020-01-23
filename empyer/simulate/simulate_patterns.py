import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import convolve
from empyer.misc.image import rotate
from empyer.misc.kernels import shape_function,sg,random_rotation,four_d_Circle
import hyperspy.api as hs

import skimage.draw as draw


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

    Assumes a spherical cluster of radius r and spots appearing at k and an Intensity of I (if the deviation parameter
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


def random_pattern(symmetry, k, radius, accept_angle=None):
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
    k = k  # +np.random.randn()/10 # normal distribution about k
    k = [[np.cos(angle*i)*k, np.sin(angle*i)*k] for i in range(symmetry)]
    rotation_vector, theta = random_rotation(acceptable_rotation_vectors=accept_angle)
    rand_angle = np.random.rand()*np.pi*2
    k = [list(rotate(x, y, rand_angle))+[0] for x, y in k]
    s = [sg(acc_voltage=200, rotation_vector=rotation_vector, theta=theta, k0=speckle) for speckle in k]
    observed_intensity = [100 * shape_function(r=radius, s=dev) for dev in s]
    return k, observed_intensity


def simulate_pattern(symmetry, k, num_clusters, probe_size, r, center, angle=None, lengths=None, acceptAngle=None):
    image = np.zeros(shape=(256, 256))
    xInd, yInd = np.mgrid[:256, :256]
    for i in range(num_clusters):
        k_val, observed_int = random_pattern(symmetry=symmetry, k=k, radius=r, accept_angle=acceptAngle)
        for pos, inten in zip(k_val, observed_int):
            circle = (xInd - pos[0] - center[0]) ** 2 + (yInd - pos[1] - center[1]) ** 2
            image[circle < probe_size] += inten
            #image[circle < probe_size] += 10
    if angle and lengths:
        image = distort(image, center, angle, lengths)
    return image


def simulate_cube(probe=2, positions=101, length=50, number_clusters=50, radius=5, accept_angle=None):
    """The general concept here is you start with a bunch of random positions for the clusters.  For all of the
    positions you then calculate the intensity of the spots and every diffraction pattern is just what patterns are
    at some postion..."""
    pos_values = range(radius, positions-radius)
    pos = list(zip(*[np.random.choice(pos_values, number_clusters),
                           np.random.choice(pos_values, number_clusters)]))
    symmetry = np.random.choice([2, 4, 6, 8, 10], number_clusters)  # random symmetry for each cluster
    patterns = [simulate_pattern(symmetry=s,
                                 k=75,
                                 num_clusters=1,
                                 r=radius,
                                 probe_size=10,
                                 center=[128, 128],
                                 angle=0,
                                 lengths=[75, 75],
                                 acceptAngle=accept_angle) for s in symmetry]
    four = np.ones((positions, positions, 256, 256))
    c = circle(radius=radius, center=pos, dim=(positions, positions))
    circlesize = np.divide(np.shape(c),2)
    for pos, pat in zip(pos, patterns):
        section = clusterConvolve(pat,c)
        index = np.array([pos[0]-circlesize[0], pos[0]+circlesize[0],pos[1]-circlesize[1],pos[1]+circlesize[1]],
                         dtype=int)
        four[index[0]:index[1],index[2]:index[3],:,:] =four[index[0]:index[1],index[2]:index[3],:,:] + section
    return four


def clusterConvolve(two_d_image,two_d_kern):
    image_shape = np.shape(two_d_image)
    section = np.repeat(two_d_kern[:, :, np.newaxis], image_shape[0], axis=2)
    section = np.repeat(section[:, :, :, np.newaxis], image_shape[1], axis=3)
    return section*two_d_image


def circle(radius, center, dim):
    """Creates a the 4d equivilent of a rod?"""
    kern = np.zeros(shape=(radius*2,radius*2))
    kern[draw.circle(r=radius, c=radius, radius=radius)] = 1
    return kern