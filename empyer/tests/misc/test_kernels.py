from unittest import TestCase
import numpy as np
import matplotlib.pyplot as plt

from empyer.misc.kernels import s_g_kernel


class TestConvert(TestCase):

    def test_s_g_kernel(self):
        k = s_g_kernel(100, 4, 10, 200)
        k.plot()
        plt.show()
        k2 = s_g_kernel(100, 4, 1, 100)
        k2.plot()
        plt.show()