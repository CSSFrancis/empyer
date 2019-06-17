from unittest import TestCase
import numpy as np

from hyperspy.signals import Signal2D, BaseSignal
from empyer.signals.diffraction_signal import DiffractionSignal
import matplotlib.pyplot as plt


class TestDiffractionSignal(TestCase):
    def setUp(self):
        d = np.random.rand(10, 512, 512)
        self.center = [276, 256]
        self.lengths = sorted(np.random.rand(2) * 100 + 100, reverse=True)
        self.angle = np.random.rand() * np.pi
        rand_angle = np.random.rand(2000) * 2 * np.pi

        rand_points = [[(np.cos(ang) * self.lengths[0]), np.sin(ang) * self.lengths[1]] for ang in rand_angle]
        rand_points = np.array([[int(point[0] * np.cos(self.angle) - point[1] * np.sin(self.angle) + self.center[0]),
                                 int(point[1] * np.cos(self.angle) + point[0] * np.sin(self.angle) + self.center[1])]
                                for point in rand_points])
        d[:, rand_points[:, 0], rand_points[:, 1]] = 10
        self.bs = BaseSignal(data=d, lazy=False)
        self.s = Signal2D(self.bs)
        self.ds = DiffractionSignal(self.s)

    def test_ellipse(self):
        self.ds.determine_ellipse()
        print(self.ds.metadata.Signal.Ellipticity.center[0])
        self.assertAlmostEqual(self.ds.metadata.Signal.Ellipticity.center[0], self.center[0], places=-1)
        self.assertAlmostEqual(self.ds.metadata.Signal.Ellipticity.center[1], self.center[1], places=-1)
        self.assertAlmostEqual(self.ds.metadata.Signal.Ellipticity.lengths[0], max(self.lengths), places=-1)
        self.assertAlmostEqual(self.ds.metadata.Signal.Ellipticity.lengths[1], min(self.lengths), places=-1)
        self.assertAlmostEqual(self.ds.metadata.Signal.Ellipticity.angle, self.angle, places=1)

    def test_conversion(self):
        converted = self.ds.calculate_polar_spectrum(phase_width=720, radius=None, parallel=False, inplace=False)
        self.assertAlmostEqual(self.ds.metadata.Signal.Ellipticity.center[0], self.center[0], places=-1)
        self.assertAlmostEqual(self.ds.metadata.Signal.Ellipticity.center[1], self.center[1], places=-1)
        self.assertAlmostEqual(self.ds.metadata.Signal.Ellipticity.lengths[0], max(self.lengths), places=-1)
        self.assertAlmostEqual(self.ds.metadata.Signal.Ellipticity.lengths[1], min(self.lengths), places=-1)
        self.assertAlmostEqual(self.ds.metadata.Signal.Ellipticity.angle, self.angle, places=1)
        self.assertLess((converted.sum(axis=(0, 1)).data > 5000).sum(), 10)

    def test_parallel_conversion(self):
        converted = self.ds.calculate_polar_spectrum(phase_width=720,
                                         radius=None,
                                         parallel=True,
                                         inplace=False)
        self.assertAlmostEqual(self.ds.metadata.Signal.Ellipticity.center[0], self.center[0], places=-1)
        self.assertAlmostEqual(self.ds.metadata.Signal.Ellipticity.center[1], self.center[1], places=-1)
        self.assertAlmostEqual(self.ds.metadata.Signal.Ellipticity.lengths[0], max(self.lengths), places=-1)
        self.assertAlmostEqual(self.ds.metadata.Signal.Ellipticity.lengths[1], min(self.lengths), places=-1)
        self.assertAlmostEqual(self.ds.metadata.Signal.Ellipticity.angle, self.angle, places=1)
        self.assertLess((converted.sum(axis=(0, 1)).data > 5000).sum(), 10)

    def test_conversion_and_mask(self):
        self.ds.masig[240:260, 0:256] = True
        converted = self.ds.calculate_polar_spectrum(phase_width=720,
                                                     radius=None,
                                                     parallel=False,
                                                     inplace=False)
        self.assertAlmostEqual(self.ds.metadata.Signal.Ellipticity.center[0], self.center[0], places=-1)
        self.assertAlmostEqual(self.ds.metadata.Signal.Ellipticity.center[1], self.center[1], places=-1)
        self.assertAlmostEqual(self.ds.metadata.Signal.Ellipticity.lengths[0], max(self.lengths), places=-1)
        self.assertAlmostEqual(self.ds.metadata.Signal.Ellipticity.lengths[1], min(self.lengths), places=-1)
        self.assertAlmostEqual(self.ds.metadata.Signal.Ellipticity.angle, self.angle, places=1)
        self.assertLess((converted.sum(axis=(0, 1)).data > 5000).sum(), 10)

