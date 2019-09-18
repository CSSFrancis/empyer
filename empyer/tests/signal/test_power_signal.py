from unittest import TestCase
import numpy as np
import matplotlib.pyplot as plt
from empyer.signals.power_signal import PowerSignal


class TestPowerSignal(TestCase):
    def setUp(self):
        setup_array = np.ones(shape=(10, 10, 20, 90))
        self.ps = PowerSignal(setup_array)
        self.ps.set_axes(3,
                         name="k",
                         scale=.1,
                         units='nm^-1')  # setting up the k axis to test fancy slicing

    def test_i_vs_k(self):
        ik = self.ps.get_i_vs_k()
        np.testing.assert_array_almost_equal(ik.data, np.ones(shape=20)*90*10*10)
        ik = self.ps.get_i_vs_k(symmetry=10)
        np.testing.assert_array_almost_equal(ik.data, np.ones(shape=20)*10*10)
        ik = self.ps.get_i_vs_k(symmetry=[8,9,10])
        np.testing.assert_array_almost_equal(ik.data, np.ones(shape=20) * 3 * 10 * 10)

    def test_get_map(self):
        print(self.ps.axes_manager)
        mapped = self.ps.get_map()
        mapped.plot()
        plt.show()
        np.testing.assert_array_almost_equal(mapped.data, np.ones(shape=(10, 10))*720*180)

    def test_lazy(self):
        print(self.ps.as_lazy())
