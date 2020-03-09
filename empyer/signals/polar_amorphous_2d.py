import numpy as np
from empyer.signals.correlation_signal import CorrelationSignal
from empyer.signals.amorphous2d import Amorphous2D
from empyer.misc.angular_correlation import angular_correlation
from hyperspy.utils import stack
from hyperspy._signals.lazy import LazySignal


class PolarAmorphous2D(Amorphous2D):
    _signal_type = "polar_signal"

    def __init__(self, *args, **kwargs):
        """Create a  Polar Signal from a numpy array.
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
        Amorphous2D.__init__(self, *args, **kwargs)
        self.metadata.set_item("Signal.type", "polar_signal")

    def as_lazy(self, *args, **kwargs):
        res = super().as_lazy(*args, **kwargs)
        res.__class__ = LazyPolarSignal
        res.__init__(**res._to_dictionary())
        return res

    def to_correlation(self, binning_factor=1, cut=0, normalize=True):
        """Create a Correlation Signal from a numpy array.

        Parameters
        ----------
        binning_factor : int
            Binning factor to speed up calculations
        cut : int or float
            The number of pixels or distance to cut off image
        normalize : boolean
            normalize with autocorrelation
        Returns
        ----------
        corr : CorrelationSignal

        """
        if isinstance(cut, float):
            cut = self.axes_manager.signal_axes[1].value2index(cut)
        self.add_mask()
        if isinstance(self.data, np.ma.masked_array):
            mask = np.reshape(self.data.mask, newshape=(-1, *reversed(self.axes_manager.signal_shape)))
        correlation = self._map_iterate(angular_correlation,
                                        iterating_kwargs=(('mask', mask),),
                                        binning=binning_factor,
                                        cut_off=cut,
                                        normalize=normalize,
                                        inplace=False)
        passed_meta_data = self.metadata.as_dictionary()
        corr = CorrelationSignal(correlation, metadata=passed_meta_data)
        shift = cut // binning_factor
        corr.axes_manager.navigation_axes = self.axes_manager.navigation_axes
        corr.set_axes(-2,
                      name="Radians",
                      scale=self.axes_manager[-2].scale*binning_factor,
                      units="rad")
        offset = shift * self.axes_manager[-1].scale*binning_factor
        corr.set_axes(-1,
                      name="k",
                      scale=self.axes_manager[-1].scale*binning_factor,
                      units=self.axes_manager[-1].units,
                      offset=offset)
        return corr

    def get_variance(self, version="omega", indicies=None):
        """Calculated the variance among some image

        Parameters
        ----------
        version : str
            The name of the FEM equation to use. 'rings' calculates the mean of the variances of all the patterns at
            some k.  'omega' calculates the variance of the annular means for every value of k.
        patterns: indicies
            Calculates the FEM pattern using only some of the patterns based on their indexes
        """

        if version is "intrapattern":
            if indicies:
                var = stack([self.inav[ind] for ind in indicies])
                annular_mean = var.nanmean(axis=-2)
                annular_mean_squared = annular_mean.nanmean() ** 2
                v = (annular_mean ** 2).nanmean()
                int_vs_k = (annular_mean_squared / v) - 1
            else:
                with self.unfolded(unfold_navigation=True, unfold_signal=False):
                    annular_mean = self.nanmean(axis=-2)
                    annular_mean_squared = annular_mean.nanmean()**2
                    v = (annular_mean ** 2).nanmean()
                    int_vs_k = (annular_mean_squared / v) - 1
                self.set_signal_type("PolarSignal")

        if version is 'innerpattern':
            if indicies:
                s = stack([self.inav[ind] for ind in indicies])
                ring_squared_average = (s ** 2).nanmean(axis=-2)
                ring_squared = s.nanmean(axis=-2) ** 2
                int_vs_k = (ring_squared_average / ring_squared) - 1
            else:
                with self.unfolded(unfold_navigation=True, unfold_signal=False):
                    ring_squared_average = (self ** 2).nanmean(axis=-2)
                    ring_squared = self.nanmean(axis=-2) ** 2
                    int_vs_k = (ring_squared_average / ring_squared) - 1
                self.set_signal_type("PolarSignal")
        int_vs_k.axes_manager[0].units = "$nm^{-1}$"
        int_vs_k.axes_manager[0].name = "k"
        return int_vs_k


class LazyPolarSignal(LazySignal, PolarAmorphous2D):

    _lazy = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)