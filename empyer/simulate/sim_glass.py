import numpy as np
from empyer.misc.image import distort
from empyer.misc.kernels import sg, shape_function
from skimage.draw import circle
from empyer.misc.image import rotate
from skimage.filters import gaussian


def random_2d_clusters(num_clusters=100, grid_size=(100, 100)):
    """Randomly place clusters at different points on a 2-D grid.
    Parameters
    ------------
    num_clusters: int
        The number of clusters in some 2-D grid
    grid_size: tuple
        The size of the grid to place the clusters on randomly.
    Returns
    -----------
    cluster_positions: array-like
        The positions of the clusters in integer positions of indexes for the grid
    cluster symmetries: array-like
        The symmetries of the clusters.  Only 2,4,6,8, and 10 allowed.
    """
    cluster_positions = list(zip(*[np.random.randint(0, grid_size[0]-1, num_clusters),
                                   np.random.randint(0, grid_size[1]-1, num_clusters)]))
    cluster_symmetries = np.random.choice([2, 4, 6, 8, 10], num_clusters)
    return cluster_positions, cluster_symmetries


def simulate_pattern(symmetry, k, radius, size=(512,512), rotation_vector=(0, 0, 1), rotation=.6, scale=10):
    """Simulates one pattern for some cluster given some k, symmetry, rotation vector and rotation about that vector.

    Parameters
    -------------
    symmetry: int
        The symmetry of the pattern
    k: float
        The k vector for the cluster. The interplanar spacing for the nano-crystal
    radius: float
        Radius of the cluster in nm.
    conv_angle: float
        The convergance angle for the experiment. Sets the width of the Ewald's sphere and the size of the speckles.
    rotation_vector: tuple
        The vecotor which describes the rotation from the beam direction
    rotation: float
        The rotation about the rotation vector for the pattern.

    Returns
    ------------
    pattern: array-like
        The pattern for the cluster
    """
    image = np.ones(size)*.01
    angle = (2*np.pi)/symmetry  # angle between speckles on the pattern
    k = [[np.cos(angle*i) * k, np.sin(angle * i) * k ]for i in range(symmetry)]  # vectors for the speckles perp to BA
    k = [list(rotate(x, y, rotation))+[0] for x, y in k]
    print(k)
    s = [sg(acc_voltage=200, rotation_vector=rotation_vector, theta=rotation, k0=speckle) for speckle in k]
    print(s)
    observed_intensity = [200 * shape_function(r=radius, s=dev) for dev in s]
    circles = [circle(int(k1[0]*scale+size[0]/2), int(k1[1]*scale+size[1]/2), radius=radius) for k1 in k]
    print(circles)
    for (r,c),i in zip(circles, observed_intensity):
        image[r, c] = i+image[r, c]
    image = gaussian(image=image, sigma=2)
    return image


def simulate_cube(probe=2, positions=101, length=50, number_clusters=50, radius=5, accept_angle=None):
    """The general concept here is you start with a bunch of random positions for the clusters.  For all of the
    positions you then calculate the intensity of the spots and every diffraction pattern is just what patterns are
    at some postion..."""
    pos_values = range(radius, positions - radius)
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
    circlesize = np.divide(np.shape(c), 2)
    for pos, pat in zip(pos, patterns):
        section = clusterConvolve(pat, c)
        index = np.array(pos[0] - circlesize[0],
                         pos[0] + circlesize[0],
                         pos[1] - circlesize[1],
                         pos[1] + circlesize[1],dtype=int)
        four[index[0]:index[1], index[2]:index[3], :, :] = four[index[0]:index[1], index[2]:index[3], :,:] + section
    return four