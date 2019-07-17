import numpy as np
from scipy.interpolate import RectBivariateSpline
from empyer.misc.image import rotate
from empyer.misc.kernels import shape_function,sg,random_rotation


def cartesian_to_ellipse(center, angle, lengths):
    xInd, yInd = np.mgrid[:512, :512]
    major = max(lengths)/np.mean(lengths)
    minor = min(lengths)/np.mean(lengths)
    xInd, yInd = xInd - center[0], yInd - center[1]
    xInd, yInd = rotate(xInd, yInd, angle=-angle)
    xInd, yInd = xInd*major, yInd*minor
    xInd, yInd = rotate(xInd, yInd, angle=angle)
    return xInd,yInd


def distort(image, center, angle, lengths):
    img_shape = np.shape(image)
    initial_y, initial_x = range(-center[1], img_shape[-2]-center[1]), range(-center[0], img_shape[-1]-center[0])
    spline = RectBivariateSpline(initial_x, initial_y, image, kx=1, ky=1)
    xInd, yInd = cartesian_to_ellipse(center=center, angle=angle, lengths=lengths)
    distorted = np.array(spline.ev(yInd, xInd))
    return distorted


def cartesian_to_ellipse(center, angle, lengths):
    xInd, yInd = np.mgrid[:512, :512]
    major = max(lengths)/np.mean(lengths)
    minor = min(lengths)/np.mean(lengths)
    xInd, yInd = xInd - center[0], yInd - center[1]
    xInd, yInd = rotate(xInd, yInd, angle=angle)
    xInd, yInd = xInd*major, yInd*minor
    xInd, yInd = rotate(xInd, yInd, angle=-angle)
    return xInd, yInd


def simulate_symmetry(symmetry=4, I=1, k=4, r=1, iterations=1000):
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
    image = np.zeros(shape=(512, 512))
    xInd, yInd = np.mgrid[:512, :512]
    for i in range(num_clusters):
        k_val, observed_int = random_pattern(symmetry=symmetry, k=k)
        for pos, inten in zip(k_val, observed_int):
            circle = (xInd - pos[0] - center[0]) ** 2 + (yInd - pos[1] - center[1]) ** 2
            image[circle < probe_size] += inten
    image = distort(image, center, angle, lengths)
    return image

