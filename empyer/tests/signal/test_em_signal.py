from unittest import TestCase
import numpy as np
import matplotlib.pyplot as plt

from hyperspy.signals import Signal2D, BaseSignal
from empyer.signals.em_signal import EM_Signal


class TestEMSignal(TestCase):
    def setUp(self):
        d = np.random.rand(10, 10, 10, 500, 700)
        d[:, :, :, 156, 256] = 10
        d[:, :, :,  356, 256] = 10
        d[:, :, :, 256, 156] = 10
        d[:, :, :,  256, 356] = 10
        d[:, :, :,  255, 356] = 10
        self.s = Signal2D(data=d)
        self.ds = EM_Signal(self.s)
        self.ds.metadata

    def test_mask_below(self):
        self.ds.plot()
        self.ds.mask_below(value=.5)
        self.ds.show_mask()

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
