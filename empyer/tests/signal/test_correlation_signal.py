from unittest import TestCase
import numpy as np

from hyperspy.signals import Signal2D
from empyer.signals.correlation_signal import CorrelationSignal


class TestPolarSignal(TestCase):
    def setUp(self):
        d = np.random.rand(10, 10, 720, 180)
        self.s = Signal2D(d)
        self.ps = CorrelationSignal(self.s)

    def test_power_spectrum(self):
        self.ps.get_power_spectrum()

