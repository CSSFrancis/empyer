import numpy as np
import hyperspy.api as hs
from numpy.random import random,choice
from empyer.misc.image import rotate
from empyer.misc.kernels import mult_quaternions
from empyer.simulate.simulate_utils import _get_rotation_matrix,_get_wavelength, _get_deviation, _shape_function
from skimage.draw import circle
from skimage.filters import gaussian

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
    def __init__(self, symmetry=10, radius=1, k=4.0, position=random(2),
                 rotation_vector=[1, 0, 0], rotation_angle=0, diffraction_intensity=600):
        """Defines a cluster with a symmetry of symmetry, a radius of radius in nm and position of position.

        Parameters:
        ----------------
        symmetry: int
            The symmetry of the cluster being simulated
        radius: float
            The radius of the cluster in nm
        position: tuple
            The position of the cluster in the simulation cube
        rotation_vector: tuple
            The vector which the cluster is rotated around
        rotation_angle: float
            The angle the cluster is rotated about
        """
        self.symmetry = symmetry
        self.radius = radius
        self.position = position
        self.rotation_vector = rotation_vector
        self.rotation_angle = rotation_angle
        self.k = k
        self.diffraction_intensity= diffraction_intensity

    def get_diffraction(self, img_size=8.0, num_pixels=512, accelerating_voltage=200):
        """Takes some image size in inverse nm and then plots the resulting

        """
        rotation_matrix = _get_rotation_matrix(self.rotation_vector, self.rotation_angle)
        sphere_radius = 1/_get_wavelength(accelerating_voltage)
        scale = (num_pixels-1)/img_size
        angle = (2 * np.pi) / self.symmetry  # angle between speckles on the pattern
        k = [[np.cos(angle * i) * self.k, np.sin(angle * i) * self.k,0] for i in
             range(self.symmetry)]  # vectors for the speckles perp to BA
        k_rotated = [np.dot(rotation_matrix, speckle) for speckle in k]
        deviation = [_get_deviation(sphere_radius,speckle) for speckle in k_rotated]
        observed_intensity = [self.diffraction_intensity/self.symmetry * _shape_function(radius=self.radius, deviation=dev)
                              for dev in deviation]
        circles = [circle(int(k1[0] * scale + num_pixels/2), int(k1[1] * scale + num_pixels/2),
                          radius=30) for k1 in k_rotated]
        image = np.ones(shape=(num_pixels,num_pixels))
        for (r, c), i in zip(circles, observed_intensity):
            image[r, c] = i + image[r, c]
        image = gaussian(image=image, sigma=2)
        return image





def get_speckle_size(accelerating_voltage=200, semi_angle=0.74, k=4.0):
    wavelength = get_wavelength(acc_voltage=accelerating_voltage)
    sphere_width = semi_angle




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


