from unittest import TestCase
import numpy as np

from hyperspy.signals import Signal2D
from empyer.signals.power_signal import PowerSignal


class TestPowerSignal(TestCase):
    def setUp(self):
        d = np.random.rand(10, 10, 720, 180)
        self.s = Signal2D(d)
        self.ps = PowerSignal(self.s)
        self.ps.set_axes(2,
                         name="k",
                         scale=.1,
                         units='nm^-1')

    def test_add_mask(self):
        self.ps.mask_shape(shape='rectangle', data=[1, 1, 1, 1])
        test_dict = [1, 1, 1, 1]
        self.assertListEqual(self.ps.metadata.Masks['test']['data'],test_dict)

    def test_get_mask(self):
        self.ps.mask_shape(shape='rectangle', data=[0, 10, 1, 10])
        self.ps.mask_shape(shape='rectangle', data=[1, 20, -10, -1])
        mask = self.ps.get_mask()
        test = np.zeros((720, 180), dtype=bool)
        test[0:10, 1:10] = 1
        test[1:20, -10:-1] = 1

    def test_i_vs_k(self):
        self.ps.get_i_vs_k()
        self.ps.get_i_vs_k(symmetry=[6])

    def test_get_map(self):
        self.ps.get_map()
        self.ps.get_map(symmetry=[6, 8, 10])
