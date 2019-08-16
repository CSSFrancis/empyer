from unittest import TestCase
import numpy as np
import hyperspy.api as hs


class TestRegisterSignal(TestCase):
    def setUp(self):
        d = np.random.rand(10, 10, 200, 720)
        d[:, :, 50, 185] = 100
        d[:, :, 50, 365] = 100
        d[:, :, 50, 545] = 100
        d[:, :, 50, 5] = 100
        self.s = hs.signals.Signal2D(d)

    def testCasting(self):
        self.s.set_signal_type("DiffractionSignal")
        self.s.determine_ellipse()
        print(hs.print_known_signal_types())
