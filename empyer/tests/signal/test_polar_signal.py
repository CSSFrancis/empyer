from unittest import TestCase
import numpy as np
import matplotlib.pyplot as plt
from hyperspy.signals import Signal2D, BaseSignal
from empyer.signals.diffraction_signal import PolarSignal


class TestPolarSignal(TestCase):
    def setUp(self):
        d = np.random.rand(10, 10, 200, 720)
        self.s = Signal2D(d)
        self.ps = PolarSignal(self.s)
        self.ps.set_axes(2,
                         name="k",
                         scale=1,
                         units='nm^-1')
        self.ps.masig[0:10, 1:10]= True

    def test_autocorrelation(self):
        self.ps.autocorrelation()

    def test_autocorrelation_mask(self):
        self.ps.masig[0:720, 0:1] = True
        self.ps.plot()
        plt.show()
        ac = self.ps.autocorrelation()
        ac2 = self.ps.autocorrelation(cut=50)
        ac2 = self.ps.autocorrelation(cut=50, binning_factor=2)

    def test_fem(self):
        self.ps.fem(version='omega')
        self.ps.fem(version='rings')
