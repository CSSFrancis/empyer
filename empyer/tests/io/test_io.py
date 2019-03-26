from unittest import TestCase
import numpy as np

from hyperspy.signals import Signal2D
from empyer.io import to_diffraction_signal


class TestIOSignal(TestCase):
    def setUp(self):
        d = np.random.rand(10, 10, 720, 180)
        self.s = Signal2D(d)

    def test_to_diffraction_signal(self):
        ds = to_diffraction_signal(self.s)
