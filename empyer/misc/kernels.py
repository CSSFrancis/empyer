import hyperspy.api as hs
import numpy as np
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
    wavelength = 1/(6.626*10**-34/np.sqrt(2*9.109*10**-31*voltage*1000*1.602*10**-19*(1+voltage/511))*10**9)
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


def s_g_kern_toAng(kern, d_hkl):
    factor = (d_hkl*np.pi/180)
    dict0 = {'size':  kern.axes_manager[0].size, 'name': 's_x', 'units': 'degrees',
             'scale': kern.axes_manager[0].scale/factor, 'offset': 0}
    dict1 = {'size': kern.axes_manager[0].size, 'name': 's_y', 'units': 'degrees',
             'scale': kern.axes_manager[0].scale/factor,'offset': 0}
    ang = hs.signals.Signal2D(data=kern.data, axes=[dict0,dict1] )
    return ang


def atomic_displacement_kernel(kernel_size, displacement_factor):
    ax = np.arange(-kernel_size // 2 + 1., kernel_size// 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    return
