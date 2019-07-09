from unittest import TestCase
import numpy as np
from empyer.misc.cartesain_to_polar import convert
from empyer.misc.image import random_ellipse
import matplotlib.pyplot as plt


class TestConvert(TestCase):
    def setUp(self):
        self.d = np.zeros((512, 512))
        self.center = [276, 256]
        self.lengths = sorted(np.random.rand(2) * 100 + 100, reverse=True)
        self.angle = np.random.rand() * np.pi
        rand_points = random_ellipse(num_points=100, center=self.center, foci=self.lengths, angle=self.angle)
        self.d[rand_points[:, 0], rand_points[:, 1]] = 10

    def test_2d_convert(self):
        conversion = convert(self.d, center=self.center, angle=self.angle, foci=self.lengths,phase_width=1440)
        s = np.sum(conversion, axis=1)
        even = np.sum(conversion, axis=0)
        #plt.plot(even)
        #plt.show()
        #self.assertLess((s > max(s)/2).sum(), 4)
        plt.imshow(conversion)
        plt.show()
        plt.imshow(self.d)
        plt.show()

