import numpy as np

from empyer.signals.em_signal import EM_Signal


class PowerSignal(EM_Signal):
    _signal_type = "diffraction_signal"

    def __init__(self, *args, **kwargs):
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

    def get_i_vs_k(self, symmetry=None):
        """ Get the intensity versus k for the summed diffraction patterns

        Parameters
        ----------
        normalize: bool
            normalize the 2-d
        Returns
        ----------
        k: array
            k values for the
        i_vs_k: intensity
        """
        if symmetry is None:
            i = self.isig[:, :].sum(axis=[0,1,2])
        else:
            i = self.isig[:, symmetry].sum(axis=[0,1,2])

        return i

    def get_map(self, k_region=[3.0, 6.0], symmetry=None):
        """Creates a 2 dimensional map of from the power spectrum.
        Parameters
        ----------
        k_region: array-like
           upper and lower k values to integrate over
        normalize: bool
            normalize the 2-d
        Returns
        ----------
        symmetry_map: 2-d array
            2 dimensional map of from the power spectrum
        """
        if symmetry is None:
            sym_map = self.isig[k_region[0]:k_region[1], :].sum(axis=[2, 3]).transpose()
        else:
            sym_map = self.isig[k_region[0]:k_region[1], symmetry].sum(axis=[2, 3]).transpose()
        return sym_map