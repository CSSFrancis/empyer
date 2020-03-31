from unittest import TestCase
import numpy as np
import matplotlib.pyplot as plt
from hyperspy.signals import Signal2D, BaseSignal
from empyer.signals.polar_amorphous2d import PolarAmorphous2D


class TestPolarSignal(TestCase):
    def setUp(self):
        d = np.ones(shape=(5, 5, 20, 90))
        d[:, :, 5, 15] = 100
        d[:, :, 5, 45] = 100
        d[:, :, 5, 75] = 100  # setting up three-fold symmetry
        self.s = Signal2D(d)
        self.ps = PolarAmorphous2D(self.s)
        self.ps.set_axes(3,
                         name="k",
                         scale=1,
                         units='nm^-1')

    def test_autocorrelation(self):
        ac = self.ps.to_correlation()
        self.assertGreater(ac.data[1, 1, 5, 30], 17)
        self.assertLess(ac.data[1, 1, 5, 29], .1)


    def test_fem_omega(self):
        fem_results = self.ps.fem(version='intra')
        self.assertAlmostEqual(np.sum(fem_results.data), 0)

    def test_fem_rings(self):
        self.ps.fem(version='rings').plot()


    def test_fem_with_filter(self):
        vari = self.ps.get_variance(version="intrapattern", indicies=[[1, 1], [1, 2], [1, 3], [2, 3]])
        self.ps.plot()
        plt.show()
        vari.plot()
        plt.show()
        self.ps.get_variance(version="innerpattern")

    def test_lazy(self):
        print(self.ps.as_lazy())
