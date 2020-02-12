from unittest import TestCase
import numpy as np
import matplotlib.pyplot as plt
import hyperspy.api as hs

from empyer.simulate.simStem import Cluster


class TestSTEMSim(TestCase):
    def setUp(self):
        self.c = Cluster(symmetry=10,radius=1, k =4.0, position=(1,1), rotation_vector=[1, 0, 0], rotation_angle=0)

    def test_get_diffraciton(self):
        self.c.get_diffraction()




