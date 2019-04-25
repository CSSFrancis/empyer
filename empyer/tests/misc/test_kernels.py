from unittest import TestCase
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import numpy as np
import matplotlib.pyplot as plt

from empyer.misc.kernels import s_g_kernel, s_g_kern_toAng


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