from unittest import TestCase
import numpy as np
import matplotlib.pyplot as plt
import hyperspy.api as hs

from empyer.simulate.simulate_patterns import simulate_pattern, simulate_symmetry, cartesian_to_ellipse, random_pattern, simulate_cube
from empyer.misc.ellipse_analysis import solve_ellipse
from empyer.misc.cartesain_to_polar import convert
from empyer.misc.angular_correlation import angular_correlation,power_spectrum


class TestSimulations(TestCase):
    def test_simulate_cube(self):
        data = simulate_cube(accept_angle=.1)
        s = hs.signals.Signal2D(data)
        s.plot()
        plt.show()

    def test_simulate_symmetry(self):
        sim = simulate_symmetry(symmetry=6, iterations=10000)
        plt.plot(sim[0])
        plt.show()
        power_sim = np.mean([np.fft.fft(s).real ** 2 for s in sim], axis=0)
        plt.plot(power_sim)
        plt.show()

    def test_simulation(self):
        p =random_pattern(4, 4)

    def test_simulate_image(self):
        i = simulate_pattern(4, k=100, num_clusters=100, probe_size=10, center=[256, 256], angle=0, lengths=[10, 10])
        plt.imshow(i)
        plt.show()

    def test_cartesian_to_ellipse(self):
        x, y = cartesian_to_ellipse(center=[256, 256], angle=np.pi /4, lengths=[1.5, 1])
        plt.scatter(x, y)
        plt.show()

    def test_simulation_correction(self):
        center = [263, 274]
        angle = np.pi/3
        lengths =[11,10]
        i = simulate_pattern(4,
                             k=100,
                             num_clusters=300,
                             probe_size=20,
                             center=center,
                             angle=angle,
                             lengths=lengths)
        i = np.random.poisson(i)
        c, l, a = solve_ellipse(i)
        # asserting the ellipse is correctly identified
        self.assertAlmostEqual(center[0], c[0], places=-1)
        self.assertAlmostEqual(center[1], c[1], places=-1)
        self.assertAlmostEqual(l[0]/l[1], max(lengths)/min(lengths), places=-1)
        self.assertAlmostEqual(angle, a, places=1)
        # converting to polar coordinates
        conversion = convert(img=i, center=c, lengths=l, angle=a)
        ac = angular_correlation(conversion, normalize=True)
        ps = power_spectrum(ac)
        #plt.imshow(i)
        #plt.show()
        plt.imshow(conversion[:, 0:359])
        plt.figure()
        plt.imshow(conversion[:, 360:719])
        plt.show()
        #plt.imshow(ac)
        plt.imshow(ps[:, 0:12])
        plt.show()


