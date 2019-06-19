from unittest import TestCase
import numpy as np
from empyer.misc.cartesain_to_polar import convert


class TestConvert(TestCase):
    def setUp(self):
        self.d = np.random.rand(512, 512)
        self.center = [276, 256]
        self.lengths = sorted(np.random.rand(2) * 100 + 100, reverse=True)
        self.angle = np.random.rand() *  np.pi
        rand_angle = np.random.rand(2000) * 2 * np.pi
        rand_points = [[(np.cos(ang) * self.lengths[0]), np.sin(ang) * self.lengths[1]] for ang in rand_angle]
        rand_points = np.array([[int(point[0] * np.cos(self.angle) - point[1] * np.sin(self.angle) + self.center[0]),
                                 int(point[1] * np.cos(self.angle) + point[0] * np.sin(self.angle) + self.center[1])]
                                for point in rand_points])
        self.d[rand_points[:, 0], rand_points[:, 1]] = 10

    def test_2d_convert(self):
        conversion = convert(self.d, center=self.center, angle=self.angle, foci=self.lengths)
        s = np.sum(conversion, axis=1)
        self.assertLess((s > max(s)/2).sum(), 4)
