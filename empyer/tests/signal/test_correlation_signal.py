from unittest import TestCase
import numpy as np

from hyperspy.signals import Signal2D
from empyer.signals.correlation_signal import CorrelationSignal


class TestPolarSignal(TestCase):
    def setUp(self):
        d = np.random.rand(10, 10, 720, 180)
        self.s = Signal2D(d)
        self.cs = CorrelationSignal(self.s)

    def test_add_mask(self):
        self.cs.add_mask(name='test', type='rectangle', data=[1, 1, 1, 1])
        test_dict = [1, 1, 1, 1]
        self.assertListEqual(self.ps.metadata.Masks['test']['data'],test_dict)

    def test_get_mask(self):
        self.cs.add_mask(name='test', type='rectangle', data=[0, 10, 1, 10])
        self.cs.add_mask(name='test1', type='rectangle', data=[1, 20, -10, -1])
        mask = self.ps.get_mask()
        test = np.zeros((720, 180), dtype=bool)
        test[0:10, 1:10] = 1
        test[1:20, -10:-1] = 1


    def test_power_spectrum(self):
        self.cs.get_power_spectrum()

    def test_autocorrelation_mask(self):
        self.ps.add_mask(name='test', type='rectangle', data=[0, 720, 0, 1])
        self.ps.add_mask(name='test1', type='rectangle', data=[1, 20, 0, 10])
        self.ps.autocorrelation()