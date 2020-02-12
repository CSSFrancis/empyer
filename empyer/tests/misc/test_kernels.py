from unittest import TestCase
import matplotlib.pyplot as plt
import hyperspy.api as hs
from empyer.misc.kernels import s_g_kernel,get_wavelength, sg, shape_function,four_d_Circle
import numpy as np
import skimage.draw as draw


class TestKernles(TestCase):

    def test_s_g(self):
        k = sg(acc_voltage=200, rotation_vector=[1,0,0], theta=0, k0=(4,0,0))


    def test_wavelength(self):
        self.assertAlmostEqual(2.51*10**-3, get_wavelength(200), 2)
        self.assertAlmostEqual(1.96 * 10 ** -3, get_wavelength(300), 2)

    def test_sg(self):
        print(sg(200, rotation_vector=(0, 1, 0), theta=0))

    def test_shape_function(self):
        plt.plot([shape_function(1, (i/100)+.00001)for i in range(100)])
        plt.show()

    def test_4d_circle(self):
        f =four_d_Circle(9,(256, 256))
        img = np.zeros((256, 256), dtype=np.uint8)
        img[draw.circle(4,4,5,shape=(256, 256))]=1
        hs.signals.Signal2D(f*img).plot()
        plt.show()


