from unittest import TestCase
import numpy as np
import dask.array as da
import numpy.testing as nptest
import time
import matplotlib.pyplot as plt

from empyer.signals.amorphous2d import Amorphous2D, LazyAmorphousSignal
import empyer as em

class TestLazyAmorphous2D(TestCase):
    def setUp(self):
        d = np.random.rand(8, 9, 10,11)
        d[:, :, 6, 5] = 10
        d[:, :,  7, 3] = 10
        d[:, :, 6, 4] = 10
        d[:, :,  4, 3] = 10
        d[:, :,  2, 3] = 10
        am_sig = Amorphous2D(data=d)
        self.lazy_am_sig= am_sig.as_lazy()

    def test_is_Lazy(self):
        self.assertIsInstance(self.lazy_am_sig, LazyAmorphousSignal)
        self.assertIsInstance(self.lazy_am_sig.data, da.core.Array)

    def test_mask(self):
        self.lazy_am_sig.add_mask()
        self.assertIsInstance(self.lazy_am_sig.data._meta,np.ma.core.MaskedArray)

    def test_mask_slicing(self):
        self.lazy_am_sig.isig[1,:4]=1
        self.lazy_am_sig.isig[1, 0:4] = np.ma.masked
        validation_array = np.zeros((8,9,10,11), dtype=bool)
        validation_array[:,:,0:4,1]=True
        np.testing.assert_array_equal(self.am_sig.data.compute().mask, validation_array)

    def test_mask_slicing2(self):
        self.am_sig.manav[:, 1].masig[2:5, 3:10] = True
        np.testing.assert_array_equal(self.am_sig.inav[:, 1].isig[2:5, 3:10].data.mask,
                                      np.ones((9, 7, 3), dtype=bool))



    def test_slice_mask_below(self):
        self.am_sig.manav[0:2, 0:2].mask_below(.5)
        self.assertGreater(self.am_sig.inav[0:2, 0:2].min(axis=(0, 1, 2, 3)).data, 0.5)
        self.assertLess(self.ds.min(axis=(0, 1, 2, 3)).data, 0.5)

    def test_slice_mask_below2(self):
        self.am_sig.manav[0:2, 0:2].masig[1:2, :].mask_below(.5)
        self.assertGreater(self.am_sig.inav[0:2, 0:2].isig[1:2, :].min(axis=(0, 1, 2, 3)).data, 0.5)
        self.assertLess(self.am_sig.min(axis=(0, 1, 2, 3)).data, 0.5)

    def test_mask_above(self):
        self.am_sig.mask_above(value=0.5)
        self.assertLess(self.ds.max(axis=(0, 1, 2, 3)).data, 0.5)

    def test_slice_mask_above(self):
        self.am_sig.manav[0:2, 0:2].mask_above(.5)
        self.assertLess(self.am_sig.inav[0:2, 0:2].max(axis=(0, 1, 2, 3)).data, 0.5)
        self.assertGreater(self.am_sig.max(axis=(0, 1, 2, 3)).data, 0.5)

    def test_slice_mask_above2(self):
        self.ds.manav[0:2, 0:2].masig[1:2, :].mask_above(.5)
        self.assertLess(self.ds.inav[0:2, 0:2].isig[1:2, :].max(axis=(0, 1, 2, 3)).data, 0.5)
        self.assertGreater(self.ds.max(axis=(0, 1, 2, 3)).data, 0.5)

    def test_slice_mask_where(self):
        self.ds.manav[0:2, 0:2].masig[1:5, :].mask_where(self.ds.inav[0:2, 0:2].isig[1:5, :].data == 10)
        self.assertLess(self.ds.inav[0:2, 0:2].isig[1:5, :].max(axis=(0, 1, 2, 3)).data, 1)

    def test_mask_circle_slice(self):
        self.ds.manav[0:2, 0:2].mask_circle(center=(5, 5), radius=3)
        self.assertTrue(self.ds.inav[1, 1].isig[5, 5].data.mask)
        self.assertTrue(self.ds.inav[1, 1].isig[7, 5].data.mask)
        self.assertFalse(self.ds.inav[1, 1].isig[1:2, 5].data.mask)
        self.assertFalse(self.ds.inav[1, 3].isig[5:6, 5].data.mask)

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