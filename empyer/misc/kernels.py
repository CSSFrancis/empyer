import hyperspy.api as hs
import numpy as np
import random
import matplotlib.pyplot as plt


def s_g_kernel(kernel_size, d_hkl, cluster_size, voltage):
    """
    Parameters
    ----------------
    kernel_size: int
        The size of the kernel to create
    d_hkl: float
        The interplanar spacing in n,
    cluster_size: float
        The size of the cluster being calculated.
    """
    wavelength = 1/(6.626*10**-34/np.sqrt(2*9.109*10**-31*voltage*1000*1.602*10**-19*(1+voltage/(2*511)))*10**9)
    print(wavelength)
    ax = np.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
    xx, yy = np.divide(np.meshgrid(ax, ax), kernel_size / 3 * cluster_size)
    scaling = 1/(kernel_size / 3 * cluster_size)
    sg = np.power((np.square(xx) + np.square(yy)), 0.5)
    print(d_hkl)
    #sg_1 = np.power(sg,-1)
    dot = np.subtract(2*wavelength*d_hkl, np.multiply(sg, d_hkl))
    #mag_1 = ((wavelength**2 + d_hkl**2)**.5)
    #mag_2 = np.power(np.add(np.power(np.subtract(wavelength, sg)**2,2), d_hkl**2),.5)
    #mag = np.multiply(mag_1, mag_2)
    #ang = np.divide(dot, mag)
    #d = np.subtract(wavelength, sg)
    #mag = np.power(np.subtract(wavelength**2, np.multiply(cos)),.5)
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


def s_g_kern_to_ang(kern, d_hkl):
    factor = (d_hkl*np.pi/180)
    dict0 = {'size':  kern.axes_manager[0].size, 'name': 's_x', 'units': 'degrees',
             'scale': kern.axes_manager[0].scale/factor, 'offset': 0}
    dict1 = {'size': kern.axes_manager[0].size, 'name': 's_y', 'units': 'degrees',
             'scale': kern.axes_manager[0].scale/factor, 'offset': 0}
    ang = hs.signals.Signal2D(data=kern.data, axes=[dict0, dict1])
    return ang


def atomic_displacement_kernel(kernel_size, displacement_factor):
    ax = np.arange(-kernel_size // 2 + 1., kernel_size// 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    return


# Functions for simulating rotations
def random_rotation(acceptable_rotation_vectors=None):
    if not acceptable_rotation_vectors:
        u = random.uniform(0, 1)
        v = random.uniform(0, 1)
        p = random.uniform(0, 1)
        alpha = 2 * np.pi * u
        beta = np.arccos(2 * v -1)
        rotation_vector = (np.cos(alpha)*np.sin(beta), np.sin(beta), np.sin(alpha)*np.cos(beta))
        theta = 2 * np.pi * p
    else:
        while True:
            u = random.uniform(0, 1)
            v = random.uniform(0, 1)
            p = random.uniform(0, 1)
            alpha = 2 * np.pi * u
            beta = np.arccos(2 * v - 1)
            rotation_vector = (np.cos(alpha) * np.sin(beta), np.sin(beta), np.sin(alpha) * np.cos(beta))
            theta = 2 * np.pi * p
            if angle_between([1, 1, 1] ,rotation_vector) < acceptable_rotation_vectors:
                print(rotation_vector,angle_between([1, 1, 1] ,rotation_vector))
                break
    return rotation_vector, theta


def sg(acc_voltage, rotation_vector, theta, k0=(4,0,0)):
    """
    Parameters
    ----------------
    acc_voltage: int
        In kV the voltage of the instrument for calculating Ewald's sphere
    rotation_vector: list
        The vector that the system is rotated around
    theta: float
        The angle of rotation
    k0: tuple
        The (x,y,z) of the original s value from the optic axis.
    """
    es_radius = 1/get_wavelength(acc_voltage)

    q1 = np.array([0, k0[0], k0[1], k0[2]])
    q2 = np.array([np.cos(theta/2),
                   rotation_vector[0]*np.sin(theta/2),
                   rotation_vector[1]*np.sin(theta/2),
                   rotation_vector[2]*np.sin(theta/2)])
    q2_conj = np.array([q2[0], -1*q2[1], -1*q2[2], -1*q2[3]])

    q3 = mult_quaternions(mult_quaternions(q2,q1),q2_conj)
    q3 = q3[1:]
    dist = np.sqrt(q3[0]**2+q3[1]**2+(-es_radius - q3[2])**2)
    s = dist-es_radius
    return s


def shape_function(r, s,):
    C = 2*np.pi*s*r
    n = ((3*(np.sin(C)-C*np.cos(C)))/C**3)**2
    return n


def get_wavelength(acc_voltage):
    h = 6.626*10**-34
    m0 = 9.109*10**-31
    e = 1.602*10**-19
    wavelength = h/np.sqrt(2*m0*acc_voltage*1000*e*(1+acc_voltage/(2*511)))*10**9
    return wavelength


def mult_quaternions(Q1,Q2):
    w0,x0,y0,z0 = Q1   # unpack
    w1,x1,y1,z1 = Q2
    return([-x1*x0 - y1*y0 - z1*z0 + w1*w0,
            x1*w0 + y1*z0 - z1*y0 + w1*x0,
            -x1*z0 + y1*w0 + z1*x0 + w1*y0,
            x1*y0 - y1*x0 + z1*w0 +w1*z0])


def four_d_Circle(radius,img_dimensions):
    """Creates a the 4d equivilent of a rod?"""
    kern =np.zeros(shape=(int(radius)*2+1,int(radius)*2+1, img_dimensions[0], img_dimensions[1]))
    X, Y = np.ogrid[:int(radius)*2+1, :int(radius)*2+1]
    center_row, center_col = int(radius), int(radius)
    dist_from_center = (X - center_row) ** 2 + (Y - center_col)**2
    radius = radius ** 2
    masked_circle = dist_from_center < radius
    kern[masked_circle,:,:] =1
    return kern


def angle_between(x,y):
    c = np.dot(x, y) / np.linalg.norm(x) / np.linalg.norm(y)  # -> cosine of the angle
    return np.arccos(c)




