import numpy as np


def random_ellipse(num_points, center, foci, angle):
    rand_angle = np.random.rand(num_points) * 2 * np.pi  # random points on a circle
    points = [[(np.cos(ang) * foci[0]), np.sin(ang) * foci[1]] for ang in rand_angle]  # circle to ellipse
    points = np.array([[round(x * np.cos(angle) - y * np.sin(angle) + center[0]),
                       round(y * np.cos(angle) + x * np.sin(angle) + center[1])]  # rotating the ellipse
                       for x, y in points], dtype=int)

    return points


def symmetrical_pattern(symmetry, center, foci, angle):
    rand_angle = np.mulitply(range(0, symmetry-1),(2 * np.pi/symmetry))
    points = [[(np.cos(ang) * foci[0]), np.sin(ang) * foci[1]] for ang in rand_angle]  # circle to ellipse
    points = np.array([[round(x * np.cos(angle) - y * np.sin(angle) + center[0]),
                       round(y * np.cos(angle) + x * np.sin(angle) + center[1])]  # rotating the ellipse
                       for x, y in points], dtype=int)

    return points