from unittest import TestCase
import numpy as np
import numpy.testing as nptest
import matplotlib.pyplot as plt

from empyer.signals.amorphous2d import Amorphous2D
from skimage.filters import gaussian


class TestAmorphous2D(TestCase):
    def setUp(self):
        d = np.random.rand(8, 9, 100,110)
        d[:, :, 60, 50] = 10
        d[:, :,  70, 30] = 10
        d[:, :, 60, 40] = 10
        d[:, :,  40, 30] = 10
        d[:, :,  20, 30] = 10
        self.am_sig = Amorphous2D(data=d)

    def test_summed(self):
        self.assertEqual(self.am_sig.sum(axis=self.am_sig.axes_manager.navigation_axes),
                         self.am_sig.metadata.Sum.nav_sum)
        self.assertEqual(self.am_sig.sum(axis=self.am_sig.axes_manager.signal_axes),
                         self.am_sig.metadata.Sum.sig_sum)

    def test_add_haadf(self):
        self.am_sig.add_haadf_intensities(np.ones((8, 9)))
        nptest.assert_array_equal(self.am_sig.metadata.HAADF.intensity, np.ones((9, 8)))
        self.assertEqual(self.am_sig.axes_manager.navigation_axes[0].scale,
                         self.am_sig.metadata.HAADF.intensity.axes_manager[0].scale)

    def test_HAADF(self):
        self.am_sig.add_haadf_intensities(np.random.normal(size=(8, 9)), 1.5, .1)

    # Testing mapping on the signal and the navigation axes
    def test_axes_map(self):
        d = np.random.rand(8, 9, 10, 11)
        a_sig = Amorphous2D(data=d)
        a_sig.inav[1, 1].isig[:, :] = np.ones(shape=(10, 11)) * 100
        def as_is(image):
            #print(image.max())
            return image
        a_sig.axis_map(function=as_is, is_navigation=True, inplace=True)
        a_sig.plot()
        plt.show()

    def test_map_skimage(self):
        d = np.random.rand(8, 9, 10, 11)
        a_sig = Amorphous2D(data=d)
        print(a_sig)
        a_sig.inav[1,1].isig[:,:]= np.ones(shape=(10,11))*100
        a_sig.plot()
        plt.show()
        a_sig.axis_map(function=gaussian, is_navigation=True, inplace=True)
        a_sig.plot()
        plt.show()

    # Tests for Masking with amorphous signal...
    def test_mask(self):
        nptest.assert_array_equal(self.am_sig.metadata.Mask.nav_mask, np.zeros((9, 8), dtype=bool))

    def test_mask_circle(self):
        # signal axis
        self.am_sig.masig.mask_circle(center=(5, 5), radius=5)
        self.assertEqual(self.am_sig.masig[5, 5], True)
        # navigation axis
        self.am_sig.manav.mask_circle(center=(5, 5), radius=5)
        self.assertEqual(self.am_sig.manav[5, 5], True)

    def test_mask_ring(self):
        # signal axis
        self.am_sig.masig.mask_rings(center=(5, 5), inner_radius=2, outer_radius=7)
        self.assertEqual(self.am_sig.masig[5, 5], False)
        # navigation axis
        self.am_sig.manav.mask_rings(center=(5, 5), inner_radius=2, outer_radius=7)
        self.assertEqual(self.am_sig.manav[5, 5], False)

    def test_sig_slice_mask(self):
        # testing using integers
        self.am_sig.masig[0:15, 0:15] = True
        self.assertEqual(self.am_sig.metadata.Mask.sig_mask[1, 1], True)
        # testing using floats
        self.am_sig.masig[0.0:15.0, 0.0:15.0] = True
        self.assertEqual(self.am_sig.metadata.Mask.sig_mask[1, 1], True)

    def test_nav_slice_mask(self):
        # testing using integers
        self.am_sig.manav[0:15, 0:15] = True
        self.assertEqual(self.am_sig.metadata.Mask.nav_mask[1, 1], True)
        # testing using floats
        self.am_sig.manav[0.0:15.0, 0.0:15.0] = True
        self.assertEqual(self.am_sig.metadata.Mask.nav_mask[1, 1], True)

    # Testing the conversion to a polar signal.
    def test_estimate_distortion(self):
        self.am_sig.estimate_distortion()
    def test_to_polar_signal(self):
        polar = self.am_sig.to_polar(inplace=False)
        print(polar)


    def test_to_lazy_signal(self):
        lazy = self.am_sig.as_lazy()
        print(lazy)


