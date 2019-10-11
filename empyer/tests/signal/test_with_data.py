from unittest import TestCase
import numpy as np
import matplotlib.pyplot as plt
from hyperspy.api import load
from empyer.misc.image import random_ellipse


class TestDiffractionSignal(TestCase):
    def setUp(self):
        file = ['/media/hdd/home/hdf5Files/row0col0.hdf5', '/media/hdd/home/hdf5Files/row0col1.hdf5']
        self.d = load(file, signal_type="DiffractionSignal", stack=True)
        self.d.masig[:1250, 885:1250] = True
        self.d.mask_border(50)
        self.d.mask_circle(center=(1050,900), radius=300)
        print(self.d)

    def test_ellipse1(self):
        self.d.determine_ellipse(num_points=20000, plot=True)
        plt.show()

    def test_elliptical_identification(self):
        self.d.determine_ellipse(num_points=2000)
        ellipse = self.d.metadata.Signal.Ellipticity
        points = random_ellipse(num_points=100, center=ellipse.center, angle=ellipse.angle, foci=ellipse.lengths)
        ps = self.d.calculate_polar_spectrum(phase_width=720, radius=[0,200])
        ps.inav[1,1].plot()
        plt.show()
        ac = ps.autocorrelation(cut=50)
        ac.sum(axis=(0, 1)).isig[:, :7.0].plot()
        plt.show()
        ac.get_summed_power_spectrum().isig[0:20, :7.0].plot(cmap='hot')
        plt.show()
        ps.sum(axis=(0, 1, 2)).plot()
        plt.show()

    def test_segmented_convert(self):
        ps = self.d.calculate_polar_spectrum(radius=[0,200], segments=2, phase_width=360)
        plt.show()
        ac = ps.autocorrelation()
        ac.get_summed_power_spectrum().isig[0:20, 3.0:7.0].plot()
        plt.show()

    def test_plot(self):

        self.d.inav[1, 1].plot()
        plt.imshow(self.d.inav[1, 1].data)
        self.d.determine_ellipse(num_points=1000)
        plt.show()

    def test_ellipse(self):
        self.d.determine_ellipse(num_points=500)

    def test_polar_unwrapping(self):
        self.d.determine_ellipse(num_points=1000)
        p = self.d.calculate_polar_spectrum()
        p.inav[1, 1].plot()
        plt.show()
        s = p.autocorrelation().get_summed_power_spectrum().isig[0:12, 3.5:6.0]
        s.plot()
        plt.show()

    def test_polar_unwrapping2(self):
        self.d.determine_ellipse()
        p = self.d.calculate_polar_spectrum()
        a = p.autocorrelation()
        a.get_summed_power_spectrum().isig[2, 3.25:5.5].plot()
        a2.get_summed_power_spectrum().isig[2, 3.25:5.5].plot()
        plt.show()

    def test_power(self):
        self.power.plot()
        plt.show()
