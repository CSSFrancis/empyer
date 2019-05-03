import numpy as np

from empyer.signals.em_signal import EM_Signal


class PowerSignal(EM_Signal):
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
        EM_Signal.__init__(self, *args, **kwargs)
        self.metadata.set_item("Signal.type", "power_signal")

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
        else:
            i = self.isig[:, symmetry].sum(axis=[0, 1, 2])

        return i

    def get_map(self, k_region=[3.0, 6.0], symmetry=None):
        # TODO: symmetry is still broken :/ I don't know why it doesn't like using arrays to slice....
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

            sym_map = self.isig[k_region[0]:k_region[1], :].sum(axis=[-1,-2]).transpose()
        else:
            print(self.isig[k_region[0]:k_region[1],:].isig[:,[5,6]])
            sym_map = self.isig[k_region[0]:k_region[1], symmetry].sum(axis=[-1, -2]).transpose()
        return sym_map
