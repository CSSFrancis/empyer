from unittest import TestCase
import numpy as np

from hyperspy.signals import Signal2D, BaseSignal
from empyer.signals.diffraction_signal import DiffractionSignal


class TestDiffractionSignal(TestCase):
    def setUp(self):
        d = np.random.rand(10, 10, 512, 512)
        d[:, :, 156, 256] = 10
        d[:, :, 356, 256] = 10
        d[:, :, 256, 156] = 10
        d[:, :, 256, 356] = 10
        d[:, :, 255, 356] = 10
        self.bs = BaseSignal(data=d, lazy=True)
        self.s = Signal2D(self.bs)
        self.ds = DiffractionSignal(self.s)
        self.ds.determine_ellipse(num_points=5)

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

    def test_conversion(self):
        self.ds.calculate_polar_spectrum(phase_width=720,
                                         radius=None,
                                         parallel=False,
                                         inplace=False)

    def test_parallel_conversion(self):
        self.ds.calculate_polar_spectrum(phase_width=720,
                                         radius=None,
                                         parallel=True,
                                         inplace=False)

    def test_conversion_and_mask(self):
        self.ds.add_mask(name='test', type='rectangle', data=[0, 10, 1, 10])
        self.ds.add_mask(name='test1', type='rectangle', data=[1, 20, -10, -1])
        self.ds.calculate_polar_spectrum(phase_width=720,
                                         radius=None,
                                         parallel=False,
                                         inplace=False)

