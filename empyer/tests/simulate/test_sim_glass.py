from unittest import TestCase
import numpy as np
import matplotlib.pyplot as plt
import hyperspy.api as hs

from empyer.simulate.sim_glass import simulate_cube, simulate_pattern, random_2d_clusters
from empyer.misc.cartesain_to_polar import to_polar_image
from empyer.misc.angular_correlation import angular_correlation,power_spectrum


class TestGlassSimulations(TestCase):
    def test_random_2d_clusters(self):
        pos, sym = random_2d_clusters()
        self.assertLessEqual(np.max(pos), 100)
        self.assertGreaterEqual(np.min(pos), 0)
        pos, sym = random_2d_clusters(grid_size=(1000, 1000))
        self.assertLessEqual(np.max(pos), 1000)
        self.assertGreaterEqual(np.min(pos), 0)

    def test_simulate_pattern(self):
        i = simulate_pattern(symmetry=12, k=12, radius=6, rotation_vector=(0, 0, 1), rotation=0.6)

    def test_polar_unwrapping(self):
        i = simulate_pattern(symmetry=6, k=12, radius=6)
        """i = i+ simulate_pattern(symmetry=4, k=12, radius=6, rotation_vector=(0, 0, 1), rotation=0.2)
        i = i+ simulate_pattern(symmetry=4, k=12, radius=6, rotation_vector=(0, 0, 1), rotation=0.7)
        i = i+ simulate_pattern(symmetry=8, k=12, radius=6, rotation_vector=(0, 0, 1), rotation=0.9)
        i = i+ simulate_pattern(symmetry=10, k=12, radius=6, rotation_vector=(0, 0, 1), rotation=0.8)
        i = i+ simulate_pattern(symmetry=4, k=12, radius=6, rotation_vector=(0, 0, 1), rotation=1.5)
        i = i+ simulate_pattern(symmetry=4, k=12, radius=6, rotation_vector=(0, 0, 1), rotation=6.4)
        i = i+ simulate_pattern(symmetry=8, k=12, radius=6, rotation_vector=(0, 0, 1), rotation=4.9)
        i = i+ simulate_pattern(symmetry=10, k=12, radius=6, rotation_vector=(0, 0, 1), rotation=2.3)
        i = i+ simulate_pattern(symmetry=10, k=12, radius=6, rotation_vector=(0, 0, 1), rotation=2.8)
        i = i+ simulate_pattern(symmetry=4, k=12, radius=6, rotation_vector=(0, 0, 1), rotation=13.5)
        i = i+ simulate_pattern(symmetry=4, k=12, radius=6, rotation_vector=(0, 0, 1), rotation=6.4)
        i = i+ simulate_pattern(symmetry=8, k=12, radius=6, rotation_vector=(0, 0, 1), rotation=5.9)
        i = i+ simulate_pattern(symmetry=10, k=12, radius=6, rotation_vector=(0, 0, 1), rotation=5.3)
        i = i+ simulate_pattern(symmetry=10, k=12, radius=6, rotation_vector=(0, 0, 1), rotation=3.8)
        i = i+ simulate_pattern(symmetry=4, k=12, radius=6, rotation_vector=(0, 0, 1), rotation=12.5)
        i = i+ simulate_pattern(symmetry=4, k=12, radius=6, rotation_vector=(0, 0, 1), rotation=3.4)
        i = i+ simulate_pattern(symmetry=8, k=12, radius=6, rotation_vector=(0, 0, 1), rotation=4.9)
        i = i+ simulate_pattern(symmetry=10, k=12, radius=6, rotation_vector=(0, 0, 1), rotation=3.8)"""
        i= i+np.random.random((512,512))*3
        plt.imshow(i)
        plt.show()
        pol = to_polar_image(i, center=(256, 256), angle=None, lengths=None, radius=[0, 200], phase_width=720)
        plt.imshow(pol)
        plt.show()
        ang = angular_correlation(pol)
        ps =power_spectrum(ang)
        plt.imshow(ang)
        plt.show()
        plt.imshow(ps[50:, 0:12], aspect=.05)
        plt.show()

    def test_polar_unwrapping(self):
        i = np.ones((512, 512))*.01
        for j in range(0, 1):
            s = np.random.choice([2,4,6,8,10])
            print(s)
            v, ro = sample_spherical_cap()
            i = i + simulate_pattern(symmetry=s, k=12, radius=6)
        i = i+ np.random.random((512, 512))*np.max(i)/4
        plt.imshow(i)
        plt.show()
        pol = to_polar_image(i, center=(256, 256), angle=None, lengths=None, radius=[0, 200], phase_width=720)
        plt.imshow(pol)
        plt.show()
        ang = angular_correlation(pol)
        ps = power_spectrum(ang)
        plt.imshow(ang)
        plt.show()
        plt.imshow(ps[50:, 0:12], aspect=.05)
        plt.show()
        plt.plot(ps[50:, 2])
        plt.plot(ps[50:, 4])
        plt.plot(ps[50:, 6])
        plt.plot(ps[50:, 8])
        plt.show()


def sample_spherical_cap(v0=(0,0,1), acceptable_angle=.3):
    while True:
        v = random_vector()
        if angle_between(v0, v) < acceptable_angle:
            print("The angle between is: ", angle_between(v0,v))
            break
    p = np.random.uniform(0, 1)
    theta = 2 * np.pi * p
    return v,theta


def random_vector():
    u = np.random.uniform(0, 1)
    v = np.random.uniform(0, 1)
    alpha = 2 * np.pi * u
    beta = np.arccos(2 * v - 1)
    return np.cos(alpha) * np.sin(beta), np.sin(beta), np.sin(alpha) * np.cos(beta)


def angle_between(x,y):
    c = np.dot(x, y) / np.linalg.norm(x) / np.linalg.norm(y)  # -> cosine of the angle

    return np.arccos(c)