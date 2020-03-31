import numpy as np


def _get_deviation(sphere_radius,k):
    """
    Parameters
    ----------------
    sphere_radius: float
        The radius of the sphere
    k0: tuple
        The (x,y,z) of the original s value from the optic axis.
    """
    dist = np.sqrt(k[0]**2+k[1]**2+(-sphere_radius - k[2])**2) # Distance from the center of sphere to k
    deviation = sphere_radius-dist # distance from edge of sphere to k
    return deviation


def _get_rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians. Using Euler-Rodrigues Formula
    """
    axis = np.asarray(axis)
    axis = axis / np.sqrt(np.dot(axis, axis))
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


def _get_wavelength(acc_voltage):
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


def _shape_function(radius, deviation):
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
    c = 2* np.pi*deviation*radius
    n = ((3*(np.sin(c)-c*np.cos(c)))/c**3)**2
    return n