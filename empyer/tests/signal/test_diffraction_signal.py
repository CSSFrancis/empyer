from unittest import TestCase
import numpy as np

from hyperspy.signals import Signal2D
from empyer.signals.diffraction_signal import DiffractionSignal


class TestBinning(TestCase):
    def setUp(self):
        d = np.random.rand(3, 3, 512, 512)
        self.s = Signal2D(data=d)
        self.ds = DiffractionSignal(self.s)

    def test_add_mask(self):
        self.ds.add_mask(name='test', type='rectangle', data=[1, 1, 1, 1])

        test_dict = [1, 1, 1, 1]
        self.assertListEqual(self.ds.metadata.Masks['test']['data'],test_dict)

    def test_get_mask(self):
        self.ds.add_mask(name='test', type='rectangle', data=[0, 10, 1, 10])
        self.ds.add_mask(name='test1', type='rectangle', data=[1, 20, -10, -1])
        mask = self.ds.get_mask()
        test = np.zeros((512,512), dtype=bool)
        test[0:10, 1:10] = 1
        test[1:20, -10:-1] = 1
        self.assertEqual(0, np.sum(np.not_equal(test, mask)))
