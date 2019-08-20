from unittest import TestCase
import numpy as np
import matplotlib.pyplot as plt
from hyperspy.signals import Signal2D, BaseSignal
from empyer.signals.diffraction_signal import PolarSignal


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

    def test_autocorrelation(self):
        ac = self.ps.autocorrelation()
        self.assertGreater(ac.data[1, 1, 50, 180],45)

    def test_autocorrelation_mask(self):
        self.ps.masig[:, 0:20] = True
        self.ps.mask_above(value=40)
        ac = self.ps.autocorrelation()
        ac.plot()
        plt.show()
        self.assertLess(ac.data[1, 1, 50, 180], 45)

    def test_autocorrelation_mask(self):
        self.ps.masig[:, 0:20] = True
        self.ps.mask_above(value=40)
        ac = self.ps.autocorrelation(cut=40.0)
        ac.plot()
        plt.show()
        self.assertLess(ac.data[1, 1, 50, 180], 45)

    def test_fem(self):
        self.ps.fem(version='omega')
        self.ps.fem(version='rings')

    def test_fem_with_filter(self):
        self.ps.add_haadf_intensities(np.random.normal(size=(10, 10)), 1.5, .1)
        vari = self.ps.fem(version="omega")
        self.ps.plot()
        plt.show()
        self.ps.fem(version="rings")

    def test_lazy(self):
        print(self.ps.as_lazy())
