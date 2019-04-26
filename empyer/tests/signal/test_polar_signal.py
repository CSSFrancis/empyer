from unittest import TestCase
import numpy as np

from hyperspy.signals import Signal2D, BaseSignal
from empyer.signals.diffraction_signal import PolarSignal


class TestPolarSignal(TestCase):
    def setUp(self):
        d = np.random.rand(10, 10, 720, 180)
        self.s = Signal2D(d)
        self.ps = PolarSignal(self.s)
        self.ps.set_axes(2,
                         name="k",
                         scale=1,
                         units='nm^-1')
        self.ps.mask_shape(shape='rectangle', data=[0, 10, 1, 10])

    def test_autocorrelation(self):
        self.ps.autocorrelation()

    def test_autocorrelation_mask(self):
        self.ps.mask_shape(shape='rectangle', data=[0, 720, 0, 1])
        self.ps.mask_shape(shape='rectangle', data=[1, 20, 0, 10])
        self.ps.autocorrelation()

    def test_fem(self):
        self.ps.fem(version='omega')
        self.ps.fem(version='rings')
