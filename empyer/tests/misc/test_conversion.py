from unittest import TestCase
import numpy as np
from empyer.misc.cartesain_to_polar import convert


class TestConvert(TestCase):
    def setUp(self):
        self.test2 = np.random.rand(10,10, 512, 512)

    def test_2d_convert(self):
        test1 = np.zeros((512, 512))
        test1[240:260, :] = 1
        c= convert(test1,
                       center=[265, 250],
                       angle=0,
                       foci=[100, 100],
                       phase_width=720)
        test1[:, 240:260] = 1
        mask = np.zeros((512, 512), dtype=bool)
        mask[:240, 240:260] = 1
        mask[260:, 240:260] = 1
        c = convert(test1,
                            mask=mask,
                            center=[265, 250],
                            angle=0,
                            foci=[100, 100],
                            phase_width=720)
        self.assertAlmostEqual(np.mean(np.subtract(c, c_mask)), 0, places=2)

    def test_nd_convert(self):
        conversion, _ = convert(self.test2)
        self.assertAlmostEqual(np.mean(conversion), 0.5, places=2)

    def test_elliptical(self):
        conversion,_ = convert(self.test2, center=[265, 250], angle=.4, foci=[100, 130], phase_width=720)
        self.assertAlmostEqual(np.mean(conversion), 0.5, places=2)
