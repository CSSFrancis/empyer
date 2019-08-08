from unittest import TestCase
import matplotlib.pyplot as plt

from empyer.misc.kernels import s_g_kernel, s_g_kern_toAng,get_wavelength, sg, shape_function


class TestConvert(TestCase):

    def test_s_g_kernel(self):
        k = s_g_kernel(100, 4, 1, 200)
        a = s_g_kern_toAng(k, 4)
        k.plot()
        a.plot()
        plt.show()
        k2 = s_g_kernel(100, 4, 20, 100)
        a2 = s_g_kern_toAng(k2, 4)
        a2.plot()

        plt.show()

    def test_wavelength(self):
        self.assertAlmostEqual(2.51*10**-3, get_wavelength(200), 2)
        self.assertAlmostEqual(1.96 * 10 ** -3, get_wavelength(300), 2)

    def test_sg(self):
        print(sg(200, rotation_vector=(0, 1, 0), theta=0))

    def test_shape_function(self):
        plt.plot([shape_function(1, (i/100)+.00001)for i in range(100)])
        plt.show()

