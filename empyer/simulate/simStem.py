import numpy as np
import hyperspy.api as hs
from numpy.random import random,choice
from empyer.misc.image import rotate
from empyer.misc.kernels import sg, shape_function
from skimage.draw import circle
import matplotlib.pyplot as plt


class SimulationCube(object):
    """Defines a simulation cube of dimensions x, y, z in nm.  This allows you to create some simulation of the cube
    based on kinematic diffraction"""
    def __init__(self, dimensions=(20, 20, 20)):
        """Initializes the simulation cube
        Parameters
        --------------
        dimensions: tuple
            The dimensions of the cube to simulate.
        """
        self.clusters = []
        self.dimensions = dimensions

    def add_random_clusters(self, num_clusters, radius_range=(.5, 1.5), k_range=(3.5, 4.5), random_rotation=True,
                            symmetry=[2, 4, 6, 8, 10]):
        """Randomly initializes the glass with a set of random clusters.
        Parameters
        ------------
        num_clusters: int
            The number of cluster to add
        radius_range: tuple
            The range of radii to randomly choose from
        k_range: tuple
            THe range of k to randomly choose from
        random_rotation: bool
            Random rotate the cluster or not
        symmetry: list
            The list of symmetries to choose from
        """
        rand_r = random(num_clusters) * (radius_range[1] - radius_range[0]) + radius_range[0]
        rand_k = random(num_clusters) * (k_range[1] - k_range[0]) + k_range[0]
        rand_sym = choice(symmetry,num_clusters)
        rand_pos = random((3, num_clusters)) * self.dimensions
        if random_rotation:
            rand_vector = random((3,num_clusters))*2-1
            rand_rot = random(num_clusters)*np.pi
        else:
            rand_vector = [1,0,0]
            rand_rot = 0
        for s, r, k, v, a in zip(rand_sym,rand_r, rand_k, rand_vector, rand_rot):
            self.clusters.append(Cluster(s,r,k,rand_pos,v,a))
        return

    def show_projection(self, acceptance, size=512):
        """Plots the 2-d projection.  Creates a 2-D projection of the clusters in the amorphous matrix.

        Parameters:
        ------------
        acceptance: float
            The angle of acceptance.  Only clusters within this projection will be allowed.
        size: int
            The size of the image made

        """
        return

    def get_4d_stem(self, convergence_angle, accelerating_voltage, simulation_size=(200,200256, 256)):
        """Returns an amorphous2d object which shows the 4d STEM projection for some set of clusters along some
        illumination

        Parameters
        ------------
        convergence_angle: float
            The convergance angle for the experiment
        accelerating_voltage: float
            The accelerating voltage for the experiment in kV
        simulation_size: tuple
            The size of the image for both the reciporical space image and the real space image.

        Returns
        ------------
        dataset: Amorphus2D
            Returns a 4 dimensional dataset which represents the cube
        """

        return dataset


class Cluster(object):
    def __init__(self, symmetry=10, radius=1, k =4.0, position=random(2), rotation_vector=[1, 0, 0], rotation_angle=0):
        """Defines a cluster with a symmetry of symmetry, a radius of radius in nm and position of position.

        Parameters:
        ----------------
        symmetry: int
            The symmetry of the cluster being simulated
        radius: float
            The radius of the cluster in nm
        position: tuple
            The position of the cluster in the simulation cube
        """
        self.symmetry = symmetry
        self.radius = radius
        self.position = position
        self.rotation_vector = [1, 0, 0]
        self.rotation_angle = rotation_angle
        self.k = k

    def get_diffraction(self, img_size=8.0, num_pixels=512, accelerating_voltage=200, scale=None):
        """Takes some image size in inverse nm and then plots the resulting
        :param img_size:
        :param accelerating_voltage:
        :param scale:
        :return:
        """
        scale = img_size/num_pixels-1
        np.ogrid[-img_size:img_size+scale:scale,-img_size:img_size+scale:scale]
        angle = (2 * np.pi) / self.symmetry  # angle between speckles on the pattern
        k = [[np.cos(angle * i) * self.k, np.sin(angle * i) * self.k] for i in
             range(self.symmetry)]  # vectors for the speckles perp to BA
        #k = [list(rotate(x, y, rotation)) + [0] for x, y in k]
        deviation_parameters = [sg(acc_voltage=200,
                                   rotation_vector=self.rotation_vector,
                                   theta=self.rotation_angle,
                                   k0=speckle)
                                for speckle in k]

        observed_intensity = [200 * shape_function(r=self.radius, s=dev) for dev in s]
        circles = [circle(int(k1[0] * scale + img_size/ 2), int(k1[1] * scale + img_size/ 2), radius=self.radius) for k1 in
                   k]
        print(circles)
        for (r, c), i in zip(circles, observed_intensity):
            image[r, c] = i + image[r, c]
        image = gaussian(image=image, sigma=2)
        return image


def get_wavelength(acc_voltage):
    """Given some accelerating voltage for a microscope calculate the relativistic wavelength
    Parameters
    -----------------
    acc_voltage: float
        The accelerating voltage of the microscope.

    Returns:
    -----------
    wavelength:float
        The wavelength of the electrons.
    """
    h = 6.626*10**-34
    m0 = 9.109*10**-31
    e = 1.602*10**-19
    wavelength = h/np.sqrt(2*m0*acc_voltage*1000*e*(1+acc_voltage/(2*511)))*10**9
    return wavelength


def shape_function(radius, s_g):
    """Returns the point at some deviation parameter s and a radius r

    Parameters
    ----------------
    r: float
        The radius of the cluster
    s_g: float
        The deviation parameter for the point.

    Returns:
    n:float
        Shape function for some deviation parameter and radius
    """
    c = 2* np.pi*s_g*radius
    n = ((3*(np.sin(c)-c*np.cos(c)))/c**3)**2
    return n


def s_g_kernel(kernel_size, d_hkl, cluster_size, voltage):
    """ Simulates a 2-d projection of the s_g kernel... (Maybe make 3-D if useful)
    Parameters
    ----------------
    kernel_size: int
        The size of the kernel to create
    d_hkl: float
        The interplanar spacing in n,
    cluster_size: float
        The size of the cluster being calculated.
    """
    wavelength = get_wavelength(acc_voltage=voltage)
    ax = np.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
    xx, yy = np.divide(np.meshgrid(ax, ax), kernel_size / 3 * cluster_size)
    scaling = 1/(kernel_size / 3 * cluster_size)
    sg = np.power((np.square(xx) + np.square(yy)), 0.5)
    print(d_hkl)
    dot = np.subtract(2*wavelength*d_hkl, np.multiply(sg, d_hkl))
    angles = np.divide(sg, d_hkl)
    sg_surf = np.multiply(sg, (2 * np.pi * cluster_size))
    kernel = np.power(np.multiply(np.divide((np.sin(sg_surf) -
                                             np.multiply(sg_surf,
                                                         np.cos(sg_surf))),
                                            np.power(sg_surf, 3)), 3), 2)
    dict0 = {'size': kernel_size, 'name': 's_x', 'units': 'nm^-1', 'scale': scaling, 'offset': 0}
    dict1 = {'size': kernel_size, 'name': 's_y', 'units': 'nm^-1', 'scale': scaling, 'offset': 0}
    k = hs.signals.Signal2D(data=kernel, axes=[dict0, dict1])
    return k
