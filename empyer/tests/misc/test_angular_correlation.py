from unittest import TestCase
import numpy as np
import matplotlib.pyplot as plt
from empyer.misc.angular_correlation import angular_correlation, power_spectrum


class TestBinning(TestCase):
    def setUp(self):
        self.test1 = np.random.rand(180, 720)
        self.mask = np.zeros((180, 720), dtype=bool)
        self.mask[0:90, 20:60] = 1

    def test_angular_correlation(self):
        ac1 = angular_correlation(self.test1)

        self.assertAlmostEqual(np.mean(ac1[:, 0]), .33, places=1)
        self.assertAlmostEqual(np.mean(ac1[:, 360]), 0, places=2)

    def test_angular_correlation_mask(self):
        test = np.ma.masked_array(np.random.rand(10, 10))
        test.mask= np.zeros((10, 10), dtype=bool)
        test.mask[0:5, 2:4] = True
        ac1 = angular_correlation(test)

    def test_power_spectrum(self):
        ac1 = angular_correlation(self.test1, normalize=False)
        ac2 = angular_correlation(self.test1, self.mask)
