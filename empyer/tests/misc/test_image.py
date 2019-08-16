from unittest import TestCase
import numpy as np
from empyer.misc.image import bin_2d, ellipsoid_list_to_cartesian, random_ellipse
import matplotlib.pyplot as plt


class TestBinning(TestCase):
    def setUp(self):
        self.test1 = np.zeros((512, 512))
        self.test2 = np.zeros((513, 513))

    def test_binning(self):
        bin1 = bin_2d(self.test1,2)
        bin2 = bin_2d(self.test2,2)
        self.assertTupleEqual(np.shape(bin1), (256,256))
        self.assertTupleEqual(np.shape(bin2), (256,256))


class TestConversions(TestCase):
    def setUp(self):
        self.thetas = np.linspace(-1*np.pi, np.pi, num=180)
        self.radii = np.arange(1, 40)

    def test_ellipse_conversion(self):
        x, y = ellipsoid_list_to_cartesian(self.radii,
                                          self.thetas,
                                          center=[0,0],
                                          axes_lengths=[10,20],
                                          angle=np.pi/4)
        plt.scatter(x, y)
        p = random_ellipse(num_points=100, center=[0,0], foci=[20,10], angle=np.pi/4)
        plt.scatter(p[:, 0], p[:, 1])
        plt.show()

    def test_rand_ellipse(self):
        p = random_ellipse(num_points=1000, center=[0,0], foci=[100,50], angle=np.pi/2)
        plt.scatter(p[:, 0], p[:, 1])
        plt.show()



