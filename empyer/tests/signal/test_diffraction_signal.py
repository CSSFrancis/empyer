from unittest import TestCase
import numpy as np

from hyperspy.signals import Signal2D, BaseSignal
from empyer.signals.diffraction_signal import DiffractionSignal, LazyDiffractionSignal
import matplotlib.pyplot as plt
from empyer.misc.image import random_ellipse
import time


class TestDiffractionSignal(TestCase):
    def setUp(self):
        d = np.random.rand(10, 10, 512, 512)
        self.center = [276, 256]
        self.lengths = sorted(np.random.rand(2) * 50 + 100, reverse=True)
        self.angle = np.random.rand() * np.pi
        rand_points = random_ellipse(num_points=1000, center=self.center, foci=self.lengths, angle=self.angle)
        d[:, :, rand_points[:, 0], rand_points[:, 1]] = 10
        self.bs = BaseSignal(data=d, lazy=False)
        self.s = Signal2D(self.bs)
        self.ds = DiffractionSignal(self.s)

    def test_ellipse(self):
        self.ds.determine_ellipse()
        print("Centers: ", self.center, self.ds.metadata.Signal.Ellipticity.center)
        print("Lengths: ", self.lengths, self.ds.metadata.Signal.Ellipticity.lengths)
        print("Angle: ", self.angle, self.ds.metadata.Signal.Ellipticity.angle)
        self.assertAlmostEqual(self.ds.metadata.Signal.Ellipticity.center[0], self.center[0], places=-1)
        self.assertAlmostEqual(self.ds.metadata.Signal.Ellipticity.center[1], self.center[1], places=-1)
        self.assertAlmostEqual(self.ds.metadata.Signal.Ellipticity.lengths[0], max(self.lengths), places=-1)
        self.assertAlmostEqual(self.ds.metadata.Signal.Ellipticity.lengths[1], min(self.lengths), places=-1)
        self.assertAlmostEqual(self.ds.metadata.Signal.Ellipticity.angle, self.angle, places=1)

    def test_conversion(self):
        start = time.time()
        converted = self.ds.calculate_polar_spectrum(phase_width=720, parallel=False, inplace=False)
        stop = time.time()
        print("the Time is:", stop-start)
        self.assertAlmostEqual(self.ds.metadata.Signal.Ellipticity.center[0], self.center[0], places=-1)
        self.assertAlmostEqual(self.ds.metadata.Signal.Ellipticity.center[1], self.center[1], places=-1)
        self.assertAlmostEqual(self.ds.metadata.Signal.Ellipticity.lengths[0], max(self.lengths), places=-1)
        self.assertAlmostEqual(self.ds.metadata.Signal.Ellipticity.lengths[1], min(self.lengths), places=-1)
        self.assertAlmostEqual(self.ds.metadata.Signal.Ellipticity.angle, self.angle, places=1)
        print("the max is:", np.max(converted.sum(axis=(0, 1, 2)).data) / 2)
        self.assertLess((converted.sum(axis=(0, 1, 2)).data > max(converted.sum(axis=(0, 1, 2)).data)/2).sum(), 5)

    def test_parallel_conversion(self):
        converted = self.ds.calculate_polar_spectrum(phase_width=720,
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
                                                     radius=[0,200],
                                                     parallel=False,
                                                     inplace=False)
        print("Centers: ", self.center, self.ds.metadata.Signal.Ellipticity.center)
        print("Lengths: ", self.lengths, self.ds.metadata.Signal.Ellipticity.lengths)
        print("Angle: ", self.angle, self.ds.metadata.Signal.Ellipticity.angle)
        self.assertAlmostEqual(self.ds.metadata.Signal.Ellipticity.center[0], self.center[0], places=-1)
        self.assertAlmostEqual(self.ds.metadata.Signal.Ellipticity.center[1], self.center[1], places=-1)
        self.assertAlmostEqual(self.ds.metadata.Signal.Ellipticity.lengths[0], max(self.lengths), places=-1)
        self.assertAlmostEqual(self.ds.metadata.Signal.Ellipticity.lengths[1], min(self.lengths), places=-1)
        self.assertAlmostEqual(self.ds.metadata.Signal.Ellipticity.angle, self.angle, places=1)
        self.assertLess((converted.sum(axis=(0, 1)).data > 5000).sum(), 10)


class TestSegmentedDiffractionSignal(TestCase):
    def setUp(self):
        self.ellipse = np.random.rand(25, 200, 200)
        self.center = [100, 110]
        self.lengths = np.add(np.random.rand(25, 2)*30, 50)
        self.lengths = [sorted(l, reverse=True) for l in self.lengths]
        self.angle = np.random.rand(25)*np.pi
        rand_angles = np.random.rand(25, 200)*2*np.pi
        rand_points = [[[(np.cos(ang)*l[0]), np.sin(ang)*l[1]] for ang in rand_angle]
                       for l, rand_angle in zip(self.lengths, rand_angles)]
        rand_points = np.array([[[int(point[0]*np.cos(a)-point[1]*np.sin(a)+self.center[0]),
                                 int(point[1] * np.cos(a) + point[0] * np.sin(a)+self.center[1])]
                                for point in rand_point]for (rand_point, a) in zip(rand_points, self.angle)])
        for i, rand_point in enumerate(rand_points):
            self.ellipse[i, rand_point[:, 0], rand_point[:, 1]] = 100
        self.ellipse = np.random.poisson(self.ellipse)
        self.ellipse = np.reshape(self.ellipse, (5,5,200,200))
        self.ellipse = np.reshape(np.transpose([[[self.ellipse]*2]*2], axes=(0,3,1,4,2,5,6)),(10,10,200,200))
        print(np.shape(self.ellipse))
        self.ellipse = np.random.poisson(self.ellipse)
        self.bs = BaseSignal(data=self.ellipse, lazy=False)
        self.s = Signal2D(self.bs)
        self.ds = DiffractionSignal(self.s)

    def test_seg(self):
        ps = self.ds.calculate_polar_spectrum(segments=5, num_points=120, radius=[0, 80])
        ps.inav[0, 1].plot()
        ps.inav[1, 1].plot()
        plt.show()

    def test_lazy(self):
        lazy = self.ds.as_lazy()
        self.assertIsInstance(lazy, LazyDiffractionSignal)
        lazy.determine_ellipse()
        lazy.mask_below(10)
        lazy.manav[2:4,1].mask_below(10)
        lazy.calculate_polar_spectrum()