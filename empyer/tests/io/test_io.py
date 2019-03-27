from unittest import TestCase
import numpy as np

from hyperspy.signals import Signal2D
from empyer.io import to_diffraction_signal, to_correlation_signal,to_polar_signal, to_power_signal


class TestIOSignal(TestCase):
    def setUp(self):
        d = np.random.rand(10, 10, 720, 180)
        self.s = Signal2D(d)

    def test_to_diffraction_signal(self):
        ds = to_diffraction_signal(self.s)
        self.assertDictEqual(ds.axes_manager.as_dictionary(), self.s.axes_manager.as_dictionary())

    def test_to_polar_signal(self):
        ps = to_polar_signal(self.s)
        self.assertDictEqual(ps.axes_manager.as_dictionary(), self.s.axes_manager.as_dictionary())

    def test_to_correlation_signal(self):
        ds = to_correlation_signal(self.s)
        self.assertDictEqual(ds.axes_manager.as_dictionary(), self.s.axes_manager.as_dictionary())

    def test_to_power_signal(self):
        ps = to_power_signal(self.s)
        self.assertDictEqual(ps.axes_manager.as_dictionary(), self.s.axes_manager.as_dictionary())
