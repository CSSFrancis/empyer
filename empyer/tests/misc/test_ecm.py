from unittest import TestCase
import numpy as np
from empyer.misc.ecm import ecm


class TestECM(TestCase):
    def setUp(self):
        self.test_series1 = np.ones(1000)
        self.test_series2 = np.ones(1000)
        every_other = np.remainder(list(range(1000)), 2) == 1
        self.test_series2[every_other] = 0

    def test_ecm(self):
        i = ecm(self.test_series1)
        self.assertListEqual(list(np.array(i, dtype=int)), list(np.ones(1000)))
        i2 = np.array(ecm(self.test_series2), dtype=int)
