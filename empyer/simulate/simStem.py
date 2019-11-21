import numpy as np
import hyperspy.api as hs


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
