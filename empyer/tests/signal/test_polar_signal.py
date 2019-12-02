from unittest import TestCase
import numpy as np
import matplotlib.pyplot as plt
from hyperspy.signals import Signal2D, BaseSignal
from empyer.signals.diffraction_signal import PolarSignal


class TestPolarSignal(TestCase):
    def setUp(self):
        d = np.ones(shape=(5, 5, 20, 90))
        d[:, :, 5, 15] = 100
        d[:, :, 5, 45] = 100
        d[:, :, 5, 75] = 100  # setting up three-fold symmetry
        self.s = Signal2D(d)
        self.ps = PolarSignal(self.s)
        self.ps.set_axes(2,
                         name="k",
                         scale=1,
                         units='nm^-1')

    def test_autocorrelation(self):
        ac = self.ps.autocorrelation()
        self.assertGreater(ac.data[1, 1, 5, 30], 17)
        self.assertLess(ac.data[1, 1, 5, 29], .1)

    def test_autocorrelation_mask(self):
        self.ps.mask_below(value=40)
        ac = self.ps.autocorrelation()
        self.assertEqual(ac.data[1, 1, 1, 1], 0)

    def test_fem(self):
        self.ps.fem(version='omega').plot()
        self.ps.fem(version='rings')

    def test_fem_with_filter(self):
        vari = self.ps.fem(version="omega", indicies=[[1, 1], [1, 2], [1, 3], [2, 3]])
        self.ps.plot()
        plt.show()
        vari.plot()
        plt.show()
        self.ps.fem(version="rings")

    def test_lazy(self):
        print(self.ps.as_lazy())
