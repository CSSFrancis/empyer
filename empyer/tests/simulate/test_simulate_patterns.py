from unittest import TestCase
import numpy as np
import matplotlib.pyplot as plt

from empyer.simulate.simulate_patterns import simulate_pattern,simulate_symmetry,cartesian_to_ellipse,random_pattern
from empyer.misc.ellipse_analysis import solve_ellipse

class TestSimulations(TestCase):
    def test_simulate_symmetry(self):
        sim = simulate_symmetry(symmetry=6, iterations=10000)
        plt.plot(sim[0])
        plt.show()
        power_sim = np.mean([np.fft.fft(s).real ** 2 for s in sim], axis=0)
        plt.plot(power_sim)
        plt.show()

    def test_simulation(self):
        random_pattern(4, 4)

    def test_simulate_image(self):
        i = simulate_pattern(4, k=100, num_clusterns=500, probe_size=20, center=[256, 256], angle=np.pi/6, lengths=[11, 10])
        c,a,l =solve_ellipse(i)
        print(c,l,a)

    def test_cartesian_to_ellipse(self):
        x, y = cartesian_to_ellipse(center=[256, 256], angle=np.pi / 4, lengths=[1.5, 1])
        plt.scatter(x, y)
        plt.show()