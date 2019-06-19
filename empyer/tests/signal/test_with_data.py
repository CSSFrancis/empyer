from unittest import TestCase
import numpy as np
import matplotlib.pyplot as plt
from empyer import load


class TestPowerSignal(TestCase):
    def setUp(self):
        file = '/home/carter/Documents/ZrCuAl(HH)4-16-2018_175W_3.8mT_175C_25s(0.83nmpsec)/hdf5_files/12.55.42 Spectrum image_pos01.hdf5'
        #file = '/home/carter/Documents/ZrCuAl(HH)4-16-2018_175W_3.8mT_175C_25s(0.83nmpsec)/hdf5_files/13.08.24 Spectrum image_pos01-2.hdf5'
        #file = '/home/carter/Documents/ZrCuAl(HH)4-16-2018_175W_3.8mT_175C_25s(0.83nmpsec)/hdf5_files/13.20.15 Spectrum image_pos02.hdf5'
        self.d = load(file).inav[0:3, 0:3]
        self.d.mask_below(300)

    def test_plot(self):
        self.d.inav[1,1].plot()
        plt.imshow(self.d.inav[1, 1].data)
        self.d.determine_ellipse(num_points=500)
        plt.show()

    def test_ellipse(self):
        self.d.determine_ellipse(num_points=500)

    def test_polar_unwrapping(self):
        self.d.determine_ellipse()
        p = self.d.calculate_polar_spectrum()
        p.inav[1, 1].plot()
        plt.show()
        s = p.autocorrelation().get_summed_power_spectrum().isig[2, 3.0:6.0]
        s.plot()
        plt.show()

    def test_polar_unwrapping2(self):
        self.d.determine_ellipse()
        p = self.d.calculate_polar_spectrum()
        a = p.autocorrelation()


        a.get_summed_power_spectrum().isig[2, 3.25:5.5].plot()
        a2.get_summed_power_spectrum().isig[2, 3.25:5.5].plot()
        plt.show()

    def test_angular_correlation(self):
        self.ac.plot()
        plt.show()

    def test_power(self):
        self.power.plot()
        plt.show()
