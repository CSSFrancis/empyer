from unittest import TestCase
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import numpy as np
import matplotlib.pyplot as plt

from empyer.misc.kernels import s_g_kernel, s_g_kern_toAng,get_wavelength, sg, shape_function,simulate_symmetry
from empyer.misc.kernels import random_pattern, simulate_pattern


class TestConvert(TestCase):

    def test_s_g_kernel(self):
        k = s_g_kernel(100, 4, 1, 200)
        a = s_g_kern_toAng(k, 4)
        k.plot()
        a.plot()
        #xx, yy = np.meshgrid(np.linspace(-21, 21, 100), np.linspace(-21, 21, 100))
        #fig = plt.figure()
        #ax = fig.gca(projection='3d')
        #ax.plot_surface(xx,yy,a.data)
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

    def test_simulate_symmetry(self):
        sim = simulate_symmetry(symmetry=6, iterations=10000)
        plt.plot(sim[0])
        plt.show()
        power_sim = np.mean([np.fft.fft(s).real**2 for s in sim], axis=0)
        plt.plot(power_sim)
        plt.show()

    def test_simulation(self):
        random_pattern(4,4)

    def test_simulate_image(self):
        i = simulate_pattern(4, 4, 100, center=[256,256], angle=0,  lengths=[10, 10])
        plt.imshow(i)
        plt.show()
