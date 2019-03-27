from unittest import TestCase
import numpy as np
import matplotlib.pyplot as plt
from empyer.misc.angular_correlation import angular_correlation, angular_power_correlation


class TestBinning(TestCase):
    def setUp(self):
        self.test1 = np.random.rand(180, 720)
        self.mask = np.zeros((180, 720), dtype=bool)
        self.mask[0:90, 20:60] = 1

    def test_angular_correlation(self):
        ac1 = angular_correlation(self.test1)
        ac2 = angular_correlation(self.test1, self.mask)
        self.assertAlmostEqual(np.mean(ac1[:, 0]), .33, places=1)
        self.assertAlmostEqual(np.mean(ac1[:, 360]), 0, places=2)
        self.assertAlmostEqual(np.mean(ac2[:, 0]), .33, places=1)
        self.assertAlmostEqual(np.mean(ac2[:, 360]), 0, places=2)

    def test_power_spectrum(self):
        ac1 = angular_correlation(self.test1, normalize=False)
        ac2 = angular_correlation(self.test1, self.mask)
        plt.imshow(angular_power_correlation(ac2))
        plt.show()
        angular_power_correlation(ac1)
