from unittest import TestCase
import numpy as np
import time
import matplotlib.pyplot as plt

from hyperspy.signals import Signal2D, BaseSignal
from empyer.signals.em_signal import EM_Signal
import empyer as em


class TestEMSignal(TestCase):
    def setUp(self):
        d = np.random.rand(8, 9, 10,11)
        d[:, :, 6, 5] = 10
        d[:, :,  7, 3] = 10
        d[:, :, 6, 4] = 10
        d[:, :,  4, 3] = 10
        d[:, :,  2, 3] = 10
        self.s = Signal2D(data=d)

        self.ds = EM_Signal(self.s)
        self.ds.save("test.hdf5", overwrite=True)
        self.ds = em.load('test.hdf5', lazy=True)
        print("done")
        self.ds.metadata

    def test_mask_slicing(self):
        self.ds.manav[(2,4,5), :].masig[1:3, 2:5] = True
        print(self.ds.data.mask)
        self.ds.plot()
        plt.show()

    def test_mask_below(self):
        self.ds.manav[(2,4,5), :].masig[1:3, :].mask_below(.5)
        print(self.ds.data.mask)

    def test_slice_mask_below(self):
        self.ds.masig[0:2, 0:2].mask_below(value=1)
        print(self.ds.data.mask)

    def test_add_mask(self):
        self.ds.mask_slice(1, 60.0, 1, 30)
        self.assertEqual(self.ds.metadata.Mask[1, 1], True)
        self.ds.mask_shape(shape='circle', data=[10, 60, 300])
        self.ds.plot()
        self.ds.show_mask()

    def test_unmask(self):
        self.ds.mask_slice(1, 3, 1, 3)
        self.ds.mask_slice(1, 3, 1, 3, unmask=True)
        self.assertEqual(self.ds.metadata.Mask[1, 1], False)

    def test_float_mask(self):
        self.ds.mask_slice(1.1, 3.1, 1.1, 3.1)
        print(self.ds.metadata.Mask)
        self.assertEqual(self.ds.metadata.Mask[2,2], True)

    def test_HAADF_mask(self):
        self.ds.add_hdaaf_intensities(np.ones((8, 9)), 1.4)
        print(self.ds.metadata.HAADF.intensity)
    def test_HAADF(self):
        self.ds.add_hdaaf_intensities(np.random.normal(size=(8, 9)), 1.5, .1)
        print(self.ds.thickness_filter())



