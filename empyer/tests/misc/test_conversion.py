from unittest import TestCase
import numpy as np
from empyer.misc.cartesain_to_polar import convert
from empyer.misc.image import random_ellipse
from timeit import timeit


class TestConvert(TestCase):
    def setUp(self):
        self.d = np.zeros((512, 512))
        self.center = [276, 256]
        self.lengths = sorted(np.random.rand(2) * 100 + 100, reverse=True)
        self.angle = np.random.rand() * np.pi
        rand_points = random_ellipse(num_points=100, center=self.center, foci=self.lengths, angle=self.angle)

        self.d[rand_points[:, 0], rand_points[:, 1]] = 10

    def test_2d_convert(self):
        start = timeit()
        conversion = convert(self.d, center=self.center, angle=self.angle, lengths=self.lengths, phase_width=720)
        stop = timeit()
        print((stop-start))
        s = np.sum(conversion, axis=1)
        even = np.sum(conversion, axis=0)
        self.assertLess((s > max(s)/2).sum(), 4)



