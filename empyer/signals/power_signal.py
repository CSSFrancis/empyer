from empyer.signals.em_signal import EMSignal
from hyperspy._signals.lazy import LazySignal
from hyperspy.signals import Signal2D, Signal1D
from hyperspy.drawing.utils import plot_images

import numpy as np


class PowerSignal(EMSignal):
    _signal_type = "power_signal"

    def __init__(self, *args, **kwargs):
        # TODO: Add in particle analysis to maps
        """Create a  Power Signal from a numpy array.
        Parameters
        ----------
        data : numpy array
           The signal data. It can be an array of any dimensions.
        axes : dictionary (optional)
            Dictionary to define the axes (see the
            documentation of the AxesManager class for more details).
        attributes : dictionary (optional)
            A dictionary whose items are stored as attributes.
        metadata : dictionary (optional)
            A dictionary containing a set of parameters
            that will to stores in the `metadata` attribute.
            Some parameters might be mandatory in some cases.
        original_metadata : dictionary (optional)
            A dictionary containing a set of parameters
            that will to stores in the `original_metadata` attribute. It
            typically contains all the parameters that has been
            imported from the original data file.
        """
        EMSignal.__init__(self, *args, **kwargs)
        self.metadata.set_item("Signal.type", "power_signal")

    def as_lazy(self, *args, **kwargs):
        """Returns the signal as a lazy signal.
        """
        res = super().as_lazy(*args, **kwargs)
        res.__class__ = LazyPowerSignal
        res.__init__(**res._to_dictionary())
        return res

    def get_i_vs_k(self, symmetry=None):
        """ Get the intensity versus k for the summed diffraction patterns

        Parameters
        ----------
        symmetry: int or array-like
            specific integers or list of symmetries to average over when creating the map of the correlations.
        Returns
        ----------
        i: Signal-2D
            The intensity as a function of k for some signal
        """
        if symmetry is None:
            i = self.isig[:, :].sum(axis=[0, 1, 2])

        elif isinstance(symmetry, int):
            i = self.isig[symmetry, :].sum()
            print(i)

        else:
            i = Signal1D(data=np.zeros(self.axes_manager.signal_shape[1]))
            for sym in symmetry:
               i = self.isig[sym, :].sum() + i
        return i

    def get_map(self, k_region=[3.0, 6.0], symmetry=None):
        """Creates a 2 dimensional map of from the power spectrum.

        Parameters
        ----------
        k_region: array-like
           upper and lower k values to integrate over, allows both ints and floats for indexing
        symmetry: int or array-like
            specific integers or list of symmetries to average over when creating the map of the correlations.
        Returns
        ----------
        symmetry_map: 2-d array
            2 dimensional map of from the power spectrum
        """
        if symmetry is None:
            sym_map = self.isig[:, k_region[0]:k_region[1]].sum(axis=[-1, -2]).transpose()

        elif isinstance(symmetry, int):
            sym_map = self.isig[symmetry, k_region[0]:k_region[1]].sum(axis=[-1]).transpose()

        else:
            sym_map = Signal2D(data=np.zeros(self.axes_manager.navigation_shape))
            for sym in symmetry:
                sym_map = self.isig[sym, k_region[0]:k_region[1]].sum(axis=[-1]).transpose() + sym_map
        return sym_map

    def plot_symmetries(self, k_region=[3.0, 6.0], symmetry=[2, 4, 6, 8, 10], *args, **kwargs):
        """Plots the symmetries in the list of symmetries. Plot symmetries takes all of the arguements that imshow does.

        Parameters
        -------------
         k_region: array-like
           upper and lower k values to integrate over, allows both ints and floats for indexing
        symmetry: list
            specific integers or list of symmetries to average over when creating the map of the correlations.
        """
        summed = [self.get_map(k_region=k_region)]
        maps = summed + [self.get_map(k_region=k_region, symmetry=i) for i in symmetries]
        l = ["summed"] + [str(i) +"-fold" for i in symmetry]
        plot_images(images=maps, label=l, *args, **kwargs)


class LazyPowerSignal(LazySignal, PowerSignal):

    _lazy = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)