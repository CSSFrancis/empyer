from unittest import TestCase
import numpy as np
from empyer.misc.image import bin_2d, ellipsoid_list_to_cartesian
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
        self.radii = np.arange(1, 180)

    def test_ellipse_conversion(self):
        x, y = ellipsoid_list_to_cartesian(self.radii,
                                          self.thetas,
                                          center=[0,0],
                                          major=20,
                                          minor=10,
                                          angle=np.pi/4,
                                          even_spaced=True)
        plt.scatter(x,y)
        plt.show()

