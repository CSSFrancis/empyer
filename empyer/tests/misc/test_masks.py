from unittest import TestCase
import numpy as np
from empyer.misc.masks import _circular_mask,_rectangular_mask,_ring_mask,_beam_stop_mask
from empyer.misc.ellipse_analysis import solve_ellipse
import matplotlib.pyplot as plt
import glob


class TestBinning(TestCase):
    def setUp(self):
        test_data = glob.glob("../test_data/*.npy")
        self.test_data_arrays = [np.load(d) for d in test_data]

    def test_beam_stop_mask(self):
        # testing the "hist" method
        for d in self.test_data_arrays:
            plt.imshow(d)
            m = _beam_stop_mask(d, method="hist")
            f = plt.figure(num=None, figsize=(20, 20), dpi=100, facecolor='w', edgecolor='k')
            plt.imshow(d)
            plt.imshow(m, alpha=.5)
            plt.show()
        return


    def test_circular_mask(self):
        m = _circular_mask(center=(10, 10), radius=5, dim=(20, 20))
        self.assertEqual(m[0, 0], False)
        self.assertEqual(m[16, 10], False)
        self.assertEqual(m[15, 10], True)

    def test_rectangular_mask(self):
        m = _rectangular_mask(x1=10, x2=15, y1=10, y2=15, dim=(20, 20))
        self.assertEqual(m[10, 10], True)

    def test_ring_mask(self):
        m = _ring_mask(center=(10, 10), outer_radius=10, inner_radius=5,  dim=(20, 20))
        self.assertEqual(m[10, 17], True)
        self.assertEqual(m[10, 10], False)

    def test_ellipse_analysis(self):
        for d in self.test_data_arrays:
            plt.imshow(d)
            mask = _beam_stop_mask(d, method="hist")
            mask = mask + _circular_mask(center=(140, 140), radius=40, dim=mask.shape)
            plt.imshow(d)
            plt.imshow(mask, alpha=.5)
            plt.show()

            c, l, a = solve_ellipse(d,mask, plot=True)
            print(c,l,a)