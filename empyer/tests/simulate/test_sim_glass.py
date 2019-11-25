from unittest import TestCase
import numpy as np
import matplotlib.pyplot as plt
import hyperspy.api as hs

from empyer.simulate.sim_glass import simulate_cube, simulate_pattern, random_2d_clusters


class TestGlassSimulations(TestCase):
    def test_random_2d_clusters(self):
        pos, sym = random_2d_clusters()
        self.assertLessEqual(np.max(pos), 100)
        self.assertGreaterEqual(np.min(pos), 0)
        pos, sym = random_2d_clusters(grid_size=(1000, 1000))
        self.assertLessEqual(np.max(pos), 1000)
        self.assertGreaterEqual(np.min(pos), 0)

    def test_simulate_pattern(self):
        i = simulate_pattern(symmetry=12,k=12,radius=6)
        plt.imshow(i)
        plt.show()
