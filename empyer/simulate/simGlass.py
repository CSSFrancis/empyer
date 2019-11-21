import numpy as np
from empyer.misc.image import distort


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
    cluster_positions = list(zip(*[np.random.randint(grid_size[0], num_clusters),
                                   np.random.randint(grid_size[1], num_clusters)]))
    cluster_symmetries = np.random.choice([2, 4, 6, 8, 10], num_clusters)
    return cluster_positions, cluster_symmetries

def simulate_pattern(symmetry, k, num_clusters, probe_size, r, center, angle=None, lengths=None, accept_angle=None):
    image = np.zeros(shape=(256, 256))
    xInd, yInd = np.mgrid[:256, :256]
    for i in range(num_clusters):
        k_val, observed_int = random_pattern(symmetry=symmetry, k=k, radius=r, accept_angle=accept_angle)
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
                         pos[1] + circlesize[1]],dtype=int)
        four[index[0]:index[1], index[2]:index[3], :, :] = four[index[0]:index[1], index[2]:index[3], :,:] + section
    return four