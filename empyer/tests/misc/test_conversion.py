from unittest import TestCase
import numpy as np
from empyer.misc.cartesain_to_polar import convert
from empyer.misc.image import random_ellipse
import time
import matplotlib.pyplot as plt


class TestConvert(TestCase):
    def setUp(self):
        self.d = np.random.random((512, 512))
        self.center = [276, 256]
        self.lengths = sorted(np.random.rand(2) * 100 + 100, reverse=True)
        self.angle = np.random.rand() * np.pi
        rand_points = random_ellipse(num_points=300, center=self.center, foci=self.lengths, angle=self.angle)

        self.d[rand_points[:, 0], rand_points[:, 1]] = 10

    def test_2d_convert(self):
        start = time.time()
        conversion, mask = convert(self.d, mask=None, radius=[0, 200], center=self.center, angle=self.angle,
                                   lengths=self.lengths, phase_width=720, normalized=False)
        stop = time.time()
        print("The time for conversion is :", stop-start)
        s = np.sum(conversion, axis=1)
        even = np.sum(conversion, axis=0)
        self.assertLess((s > max(s)/2).sum(), 120)





