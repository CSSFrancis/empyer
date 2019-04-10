from unittest import TestCase
import numpy as np

from hyperspy.signals import Signal2D, BaseSignal
from empyer.signals.diffraction_signal import DiffractionSignal


class TestDiffractionSignal(TestCase):
    def setUp(self):
        d = np.random.rand(10, 10, 512, 512)
        d[:, :, 156, 256] = 10
        d[:, :, 356, 256] = 10
        d[:, :, 256, 156] = 10
        d[:, :, 256, 356] = 10
        d[:, :, 255, 356] = 10
        self.bs = BaseSignal(data=d, lazy=True)
        self.s = Signal2D(self.bs)
        self.ds = DiffractionSignal(self.s)
        self.ds.determine_ellipse(num_points=5)

    def test_conversion(self):
        self.ds.calculate_polar_spectrum(phase_width=720,
                                         radius=None,
                                         parallel=False,
                                         inplace=False)

    def test_parallel_conversion(self):
        self.ds.calculate_polar_spectrum(phase_width=720,
                                         radius=None,
                                         parallel=True,
                                         inplace=False)

    def test_conversion_and_mask(self):
        self.ds.mask_slice(242, 262, 0, 512)
        ps = self.ds.calculate_polar_spectrum(phase_width=720,
                                              radius=None,
                                              parallel=False,
                                              inplace=False)
        ps.plot()
        ps.show_mask()
        ac = ps.autocorrelation()
        ac.plot()
        self.ds.show_mask()


