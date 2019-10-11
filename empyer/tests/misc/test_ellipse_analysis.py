from unittest import TestCase
import numpy as np
from empyer.misc.ellipse_analysis import solve_ellipse, get_max_positions
from empyer.misc.cartesain_to_polar import convert
from empyer.misc.image import random_ellipse
import matplotlib.pyplot as plt


class TestConvert(TestCase):
    def setUp(self):
        self.ellipse = np.random.rand(100, 200, 200)*90
        self.center = [110, 100]
        self.lengths = np.random.rand(100, 2)*30+50
        self.lengths = [sorted(l, reverse=True) for l in self.lengths]
        self.angles = np.random.rand(100)*np.pi
        self.rand_points = np.array([random_ellipse(num_points=1000, center=self.center, foci=l, angle=a) for
                       l, a in zip(self.lengths, self.angles)])
        for i, rand_point in enumerate(self.rand_points):
            self.ellipse[i, rand_point[:, 0], rand_point[:, 1]] = 100
        self.ellipse = np.random.poisson(self.ellipse / 100 * 10 / 10 * 100)

    def test_get_max_coords(self):
        plt.imshow(self.ellipse[0,:,:])
        c = get_max_positions(image=self.ellipse[0,:,:],radius=100, num_points=500)
        plt.scatter(c[0], c[1])
        plt.show()

    def test_solve_one_ellipse(self):
        e = self.ellipse[0, :, :]
        l = self.lengths[0]
        a = self.angles[0]
        solve_ellipse(e, num_points=1000)
        print("Length: ", l, "Angle: ", a)

    def test_solve_ellipse(self):
        for e, l, a in zip(self.ellipse, self.lengths, self.angles):
            if np.abs(l[0]-l[1]) < 1:  # too close to a circle
                continue
            cen, le, an = solve_ellipse(e, num_points=75)
            print("Centers: ", self.center, cen)
            print("Lengths: ", l, le)
            print("Angle: ", a, an)
            self.assertAlmostEqual(self.center[0], cen[0], places=-1)
            self.assertAlmostEqual(self.center[1], cen[1], places=-1)
            self.assertAlmostEqual(le[0], max(l), places=-1)
            self.assertAlmostEqual(le[1], min(l), places=-1)
            self.assertAlmostEqual(a, an, places=0)

    def test_solve_ellipse_mask(self):
        e = np.ma.masked_array(self.ellipse)
        e.mask = False
        e.mask[230:275, 0:256] = True
        e.mask[0:10, 0:256] = True
        c, l, a = solve_ellipse(e, num_points=150)
        print("Centers: ", self.center, c)
        print("Lengths: ", self.lengths, l)
        print("Angle: ", self.angles, a)
        self.assertAlmostEqual(c[0], self.center[0], places=-1)
        self.assertAlmostEqual(c[1], self.center[1], places=-1)
        self.assertAlmostEqual(l[0], max(self.lengths), places=-1)
        self.assertAlmostEqual(l[1], min(self.lengths), places=-1)
        self.assertAlmostEqual(a, self.angle, places=1)
