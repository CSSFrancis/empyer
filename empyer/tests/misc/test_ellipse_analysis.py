from unittest import TestCase
import numpy as np
from empyer.misc.ellipse_analysis import solve_ellipse
from empyer.misc.cartesain_to_polar import convert
import matplotlib.pyplot as plt


class TestConvert(TestCase):
    def setUp(self):
        self.ellipse = np.random.rand(100, 200, 200)
        self.center = [100, 110]
        self.lengths = np.random.rand(100, 2)*30+50
        self.lengths = [sorted(l, reverse=True) for l in self.lengths]
        print(self.lengths)
        self.angle = np.random.rand(100)*np.pi
        rand_angles = np.random.rand(100, 200)*2*np.pi
        rand_points = [[[(np.cos(ang)*l[0]), np.sin(ang)*l[1]] for ang in rand_angle]
                       for l, rand_angle in zip(self.lengths, rand_angles)]
        rand_points = np.array([[[int(point[0]*np.cos(a) + point[1]*np.sin(a)+self.center[0]),
                                 int(point[1] * np.cos(a) - point[0] * np.sin(a)+self.center[1])]
                                for point in rand_point]for (rand_point, a) in zip(rand_points, self.angle)])
        print(rand_points.shape)
        for i, rand_point in enumerate(rand_points):
            self.ellipse[i, rand_point[:, 0], rand_point[:, 1]] = 100
        self.ellipse = np.random.poisson(self.ellipse)

    def test_solve_ellipse(self):
        for e, l, a in zip(self.ellipse, self.lengths, self.angle):
            if np.abs(l[0]-l[1]) < 1:
                continue
            cen, le, an = solve_ellipse(e, num_points=75)
            print("Centers: ", self.center, cen)
            print("Lengths: ", l, le)
            print("Angle: ", a, an)
            self.assertAlmostEqual(self.center[0], cen[0], places=-1)
            self.assertAlmostEqual(self.center[1], cen[1], places=-1)
            #self.assertAlmostEqual(le[0], max(l), places=-1)
            #self.assertAlmostEqual(le[1], min(l), places=-1)
            self.assertAlmostEqual(a, an, places=1)

    def test_solve_ellipse_mask(self):
        e = np.ma.masked_array(self.ellipse)
        e.mask = False
        e.mask[230:275, 0:256] = True
        e.mask[0:10, 0:256] = True
        c, l, a = solve_ellipse(e, num_points=150)
        print("Centers: ", self.center, c)
        print("Lengths: ", self.lengths, l)
        print("Angle: ", self.angle, a)
        self.assertAlmostEqual(c[0], self.center[0], places=-1)
        self.assertAlmostEqual(c[1], self.center[1], places=-1)
        self.assertAlmostEqual(l[0], max(self.lengths), places=-1)
        self.assertAlmostEqual(l[1], min(self.lengths), places=-1)
        self.assertAlmostEqual(a, self.angle, places=1)
