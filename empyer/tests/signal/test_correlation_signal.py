from unittest import TestCase
import matplotlib.pyplot as plt

from empyer.io import load


class TestPolarSignal(TestCase):
    def setUp(self):
        self.ps = load('/media/hdd/home/FEMImages_Angular_analyzed/20180415_ZCA-HH-_50W_3.8mT_RT_78s-0.22nm_sec_295/15.49.20 Spectrum image_pos01_angular.hdf5')

    def test_power_spectrum(self):
        power = self.ps.get_power_spectrum()
        power.isig[2:12, 3.0:6.0].plot()
        plt.show()

    def test_summed_power_spectrum(self):
        print(self.ps)
        sum_power = self.ps.get_summed_power_spectrum()
        sum_power.isig[2:12, 3.0:6.0].plot()
        plt.show()
        sum_power.isig[6, 3.0:8.0].plot()
        plt.show()

    def test_lazy(self):
        print(self.ps.as_lazy())