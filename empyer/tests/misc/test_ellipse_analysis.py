from unittest import TestCase
import numpy as np
from empyer.misc.ellipse_analysis import solve_ellipse
from empyer.misc.cartesain_to_polar import convert
import matplotlib.pyplot as plt


class TestConvert(TestCase):
    def setUp(self):
        self.ellipse = np.random.rand(512, 512)
        self.center = [270, 256]
        self.lengths = np.random.rand(2)*100+100
        self.lengths = sorted(self.lengths, reverse=True)
        self.angle = np.random.rand()*np.pi
        rand_angle = np.random.rand(1000)*2*np.pi
        rand_points = [[(np.cos(ang)*self.lengths[0]), np.sin(ang)*self.lengths[1]] for ang in rand_angle]
        rand_points = np.array([[int(point[0]*np.cos(self.angle)-point[1]*np.sin(self.angle)+self.center[0]),
                                 int(point[1] * np.cos(self.angle) + point[0] * np.sin(self.angle)+self.center[1])]
                                for point in rand_points])
        self.ellipse[rand_points[:, 0], rand_points[:, 1]] = 100
        self.ellipse[rand_points[1, 0], rand_points[1, 1]] = 1000
        print("Point!",rand_points[1, 0], rand_points[1, 1])
        self.ellipse = np.random.poisson(self.ellipse)

    def test_solve_ellipse(self):

        c, l, a = solve_ellipse(self.ellipse, num_points=500)
        print("Centers: ", self.center, c)
        print("Lengths: ", self.lengths, l)
        print("Angle: ", self.angle, a)
        self.assertAlmostEqual(c[0], self.center[0], places=-1)
        self.assertAlmostEqual(c[1], self.center[1], places=-1)
        self.assertAlmostEqual(l[0], max(self.lengths), places=-1)
        self.assertAlmostEqual(l[1], min(self.lengths), places=-1)
        self.assertAlmostEqual(a, self.angle, places=1)

    def test_solve_ellipse_mask(self):
        e = np.ma.masked_array(self.ellipse)
        e.mask = False
        e.mask[230:275, 0:256] = True
        e.mask[0:10, 0:256] = True
        c, l, a = solve_ellipse(e, num_points=100)
        print("Centers: ", self.center, c)
        print("Lengths: ", self.lengths, l)
        print("Angle: ", self.angle, a)
        self.assertAlmostEqual(c[0], self.center[0], places=-1)
        self.assertAlmostEqual(c[1], self.center[1], places=-1)
        self.assertAlmostEqual(l[0], max(self.lengths), places=-1)
        self.assertAlmostEqual(l[1], min(self.lengths), places=-1)
        self.assertAlmostEqual(a, self.angle, places=1)
