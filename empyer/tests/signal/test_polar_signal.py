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

    def test_autocorrelation(self):
        self.ps.autocorrelation()

    def test_autocorrelation_mask(self):
        self.ps.masig[:, 0:20] = True
        self.ps.plot()
        plt.show()
        ac = self.ps.autocorrelation()
        ac.plot()
        plt.show()

    def test_fem(self):
        self.ps.fem(version='omega')
        self.ps.fem(version='rings')

    def test_fem_with_filter(self):
        self.ps.add_haadf_intensities(np.random.normal(size=(10, 10)), 1.5, .1)
        vari = self.ps.fem(version="omega")
        self.ps.plot()
        plt.show()
        self.ps.fem(version="rings")
