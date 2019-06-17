from unittest import TestCase
import numpy as np
import matplotlib.pyplot as plt
import empyer as em


class TestPowerSignal(TestCase):
    def setUp(self):
        file = '/media/hdd/home/Zr65Cu27.5Al7.5FEM/HighTDatasets/Zr65Cu27.5Al7.5_1.19nmpsec(300W_3.8mT_170C_13sec)/19.38.03 Spectrum image_pos01.hdf5'
        file2 = '/media/hdd/home/Zr65Cu27.5Al7.5FEM/HighTDatasets/Zr65Cu27.5Al7.5_1.19nmpsec(300W_3.8mT_170C_13sec)/19.50.51 Spectrum image_pos01.hdf5'

        self.d = em.load(file)
        self.d.mask_below(300)
        self.d2 = em.load(file2)
        self.d2.mask_below(300)

    def test_ellipse(self):
        self.d.determine_ellipse(num_points=500)

    def test_polar_unwrapping(self):
        p = self.d.calculate_polar_spectrum()
        print(p)
        p.plot()
        plt.show()

    def test_polar_unwrapping(self):
        self.d.determine_ellipse()
        p = self.d.calculate_polar_spectrum()
        a = p.autocorrelation()

        self.d2.determine_ellipse()
        p2 = self.d2.calculate_polar_spectrum()
        a2 = p2.autocorrelation()

        a.get_summed_power_spectrum().isig[2, 3.25:5.5].plot()
        a2.get_summed_power_spectrum().isig[2, 3.25:5.5].plot()
        plt.show()

    def test_angular_correlation(self):
        self.ac.plot()
        plt.show()

    def test_power(self):
        self.power.plot()
        plt.show()
