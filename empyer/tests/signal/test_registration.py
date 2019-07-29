from unittest import TestCase
import numpy as np
import matplotlib.pyplot as plt
#from empyer.signals.diffraction_signal import PolarSignal
import hyperspy.api as hs


class TestPolarSignal(TestCase):
    def setUp(self):
        d = np.random.rand(10, 10, 200, 720)
        d[:, :, 50, 185] = 100
        d[:, :, 50, 365] = 100
        d[:, :, 50, 545] = 100
        d[:, :, 50, 5] = 100
        self.s = Signal2D(d)
        self.ps = PolarSignal(self.s)
        self.ps.set_axes(2,
                         name="k",
                         scale=1,
                         units='nm^-1')

    def test_reg(self):
        print(hs.print_known_signal_types())
