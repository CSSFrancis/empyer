from unittest import TestCase
import numpy as np
from empyer.misc.ellipse_analysis import solve_ellipse


class TestConvert(TestCase):
    def setUp(self):
        self.ellipse = np.random.rand(512, 512)
        self.ellipse[156, 256] = 10
        self.ellipse[356, 256] = 10
        self.ellipse[256, 156] = 10
        self.ellipse[256, 356] = 10
        self.ellipse[255, 356] = 10

    def test_solve_ellipse(self):
        center, lengths, angle = solve_ellipse(self.ellipse, num_points=5)
        self.assertAlmostEqual(center[0],256, places=2)
        self.assertAlmostEqual(center[0],256, places=2)
        self.assertAlmostEqual(lengths[0],100, places=-1)
        self.assertAlmostEqual(angle, np.pi/4, places=2)
