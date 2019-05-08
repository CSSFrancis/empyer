from unittest import TestCase
import numpy as np
import matplotlib.pyplot as plt
import empyer as em


class TestPowerSignal(TestCase):
    def setUp(self):
        self.d = em.load('/media/hdd/home/FEMImages_Angular_analyzed/20170328_ZCA(HH)_50W_3.8mT_170C_78s/ZrCuAl_50W_3.hdf5')
        self.d.mask_below(200)
        print(self.d)
        self.ps = self.d.calculate_polar_spectrum()
        self.ps.save('/media/hdd/home/FEMImages_Angular_analyzed/20170328_ZCA(HH)_50W_3.8mT_170C_78s/ZrCuAl_50W_3_polar.hdf5',overwrite=True)
        self.ac = self.ps.autocorrelation()
        self.ac.save('/media/hdd/home/FEMImages_Angular_analyzed/20170328_ZCA(HH)_50W_3.8mT_170C_78s/ZrCuAl_50W_3_angular.hdf5',overwrite=True)
        self.power = self.ac.get_power_spectrum()
        self.power.save('/media/hdd/home/FEMImages_Angular_analyzed/20170328_ZCA(HH)_50W_3.8mT_170C_78s/ZrCuAl_50W_3_angularPower.hdf5',overwrite=True)

    def test_mask(self):
        self.d.show_mask()

    def test_polar_unwrapping(self):
        self.ps.plot()
        plt.show()

    def test_angular_correlation(self):
        self.ac.plot()
        plt.show()

    def test_power(self):
        self.power.plot()
        plt.show()
