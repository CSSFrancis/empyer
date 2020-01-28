from unittest import TestCase
import numpy as np
import numpy.testing as nptest
import time
import matplotlib.pyplot as plt

from empyer.signals.amorphous2d import Amorphous2D
import empyer as em


class TestAmorphous2D(TestCase):
    def setUp(self):
        d = np.random.rand(8, 9, 10,11)
        d[:, :, 6, 5] = 10
        d[:, :,  7, 3] = 10
        d[:, :, 6, 4] = 10
        d[:, :,  4, 3] = 10
        d[:, :,  2, 3] = 10
        self.am_sig = Amorphous2D(data=d)

    def test_mask(self):
        self.assertEqual(self.am_sig.metadata.Mask.nav_mask, False)
        self.assertEqual(self.am_sig.metadata.Mask.sig_mask, False)

    def test_mask_circle_int(self):
        self.am_sig.mask_circle(center=(5, 5), radius=5)

    def test_add_haadf(self):
        print(self.am_sig.metadata)
        self.am_sig.add_haadf_intensities(np.ones((8, 9)))
        nptest.assert_array_equal(self.am_sig.metadata.HAADF.intensity, np.ones((9, 8)))
        self.assertEqual(self.am_sig.axes_manager.navigation_axes[0].scale,
                         self.am_sig.metadata.HAADF.intensity.axes_manager[0].scale)

    def test_HAADF(self):
        self.ds.add_hdaaf_intensities(np.random.normal(size=(8, 9)), 1.5, .1)

    def test_lazy_signal(self):
        lazy = self.ds.as_lazy()
        print(lazy)


