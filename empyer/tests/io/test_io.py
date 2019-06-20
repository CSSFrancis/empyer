from unittest import TestCase
import numpy as np

from hyperspy.signals import Signal2D
from empyer.io import load, to_diffraction_signal, to_correlation_signal,to_polar_signal, to_power_signal


class TestIOSignal(TestCase):
    def setUp(self):
        d = np.random.rand(10, 10, 720, 180)
        self.s = Signal2D(d)
        self.ds = to_diffraction_signal(self.s)

    def test_to_diffraction_signal(self):
        ds = to_diffraction_signal(self.s)
        self.assertDictEqual(ds.axes_manager.as_dictionary(), self.s.axes_manager.as_dictionary())
        ds.mask_below(value=.5)
        ds.save(filename='temp', overwrite=True, extension='hdf5')
        ds_2 = load('temp.hdf5')
        self.assertEqual(ds.metadata.Signal.signal_type, ds_2.metadata.Signal.signal_type)
        self.assertIsInstance(ds_2.data, np.ma.masked_array)

    def test_to_polar_signal(self):
        ps = to_polar_signal(self.s)
        self.assertDictEqual(ps.axes_manager.as_dictionary(), self.s.axes_manager.as_dictionary())
        ps.save(filename='temp', overwrite=True, extension='hdf5')
        ps_2 = load('temp.hdf5')
        self.assertEqual(ps.metadata.Signal.signal_type,ps_2.metadata.Signal.signal_type)

    def test_to_correlation_signal(self):
        ds = to_correlation_signal(self.s)
        self.assertDictEqual(ds.axes_manager.as_dictionary(), self.s.axes_manager.as_dictionary())
        ds.save(filename='temp', overwrite=True, extension='hdf5')
        ds_2 = load('temp.hdf5')
        self.assertEqual(ds.metadata.Signal.signal_type, ds_2.metadata.Signal.signal_type)

    def test_to_power_signal(self):
        ps = to_power_signal(self.s)
        self.assertDictEqual(ps.axes_manager.as_dictionary(), self.s.axes_manager.as_dictionary())
        ps.save(filename='temp', overwrite=True, extension='hdf5')
        ps_2 = load('temp.hdf5')
        self.assertEqual(ps.metadata.Signal.signal_type, ps_2.metadata.Signal.signal_type)

    def test_save_and_load(self):
        self.ds.save(filename='temp', overwrite=True, extension='hdf5')
        ds_2 = load('temp.hdf5')
        self.assertEqual("diffraction_signal", ds_2.metadata.Signal.signal_type)