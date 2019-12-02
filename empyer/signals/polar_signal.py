import numpy as np
from empyer.signals.em_signal import EMSignal
from empyer.signals.correlation_signal import CorrelationSignal
from empyer.misc.angular_correlation import angular_correlation
from empyer.misc.image import square
from hyperspy.utils import stack
from hyperspy._signals.lazy import LazySignal


class PolarSignal(EMSignal):
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
        EMSignal.__init__(self, *args, **kwargs)
        self.metadata.set_item("Signal.type", "polar_signal")

    def as_lazy(self, *args, **kwargs):
        res = super().as_lazy(*args, **kwargs)
        res.__class__ = LazyPolarSignal
        res.__init__(**res._to_dictionary())
        return res

    def autocorrelation(self, binning_factor=1, cut=0, normalize=True):
        # TODO: Add the ability to cutoff like slicing (maybe use np.s)
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
        angle : CorrelationSignal

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
        angular = CorrelationSignal(correlation, metadata=passed_meta_data)
        shift = cut // binning_factor
        angular.axes_manager.navigation_axes = self.axes_manager.navigation_axes
        angular.set_axes(-2,
                         name="Radians",
                         scale=self.axes_manager[-2].scale*binning_factor,
                         units="rad")
        offset = shift * self.axes_manager[-1].scale*binning_factor
        angular.set_axes(-1,
                         name="k",
                         scale=self.axes_manager[-1].scale*binning_factor,
                         units=self.axes_manager[-1].units,
                         offset=offset)
        return angular

    def correlation_lengths(self):
        """Calculates the average correlation length across the sample 
        """

    def fem(self, version="omega", indicies=None):
        """Calculated the variance among some image

        Parameters
        ----------
        version : str
            The name of the FEM equation to use. 'rings' calculates the mean of the variances of all the patterns at
            some k.  'omega' calculates the variance of the annular means for every value of k.
        patterns: indicies
            Calculates the FEM pattern using only some of the patterns based on their indexes
        """

        if version is "omega":
            if indicies:
                var = stack([self.inav[ind] for ind in indicies])
                print(var)
                v = var.map(square, inplace=False).nanmean(axis=-2)
                center = var.nanmean(axis=-2)
                center.map(square)
                center = center.nanmean()
                int_vs_k = ((v - center) / center).nanmean()
            else:
                with self.unfolded(unfold_navigation=True, unfold_signal=False):
                    v = self.map(square, inplace=False).nanmean(axis=-2)
                    center = self.nanmean(axis=-2)
                    center.map(square)
                    center = center.nanmean()
                    int_vs_k=((v - center) / center).nanmean()
                self.set_signal_type("PolarSignal")

        if version is 'rings':
            if indicies:
                var = stack([self.inav[ind] for ind in indicies])
                print(var)
                v = var.map(square, inplace=False).nanmean().nanmean(axis=-2)
                center = var.nanmean(axis=-2)
                center.map(square)
                center = center.nanmean()
                int_vs_k = ((v - center) / center)
            else:
                with self.unfolded(unfold_navigation=True, unfold_signal=False):
                    v = self.map(square, inplace=False).nanmean().nanmean(axis=-2)
                    center = self.nanmean(axis=-2)
                    center.map(square)
                    center = center.nanmean()
                    int_vs_k = ((v - center) / center)
                self.set_signal_type("PolarSignal")
        int_vs_k.axes_manager[0].units = "$nm^{-1}$"
        int_vs_k.axes_manager[0].name = "k"
        return int_vs_k
"""
        if not self.metadata.has_item('HAADF'):
            print("No thickness filter applied...")
            if version is 'rings':
                var = self.nanmean(axis=-1)
                var.map(square)
                var = var.nanmean()
                center = self.nanmean(axis=-1).nanmean()
                center.map(square)
                int_vs_k = (var - center) / center
                print(int_vs_k.axes_manager)
            elif version is 'omega':
                var = self.map(square, show_progressbar=False, inplace=False).nanmean().nanmean(axis=1)
                center = self.nanmean(axis=-1)
                center.map(square)
                center = center.nanmean()
                int_vs_k = (var - center) / center
                print(int_vs_k.axes_manager)
        else:
            filt, thickness = self.thickness_filter()
            if version is 'rings':
                int_vs_k = []
                for i, th in enumerate(thickness):
                    index = np.where(filt.transpose() == i+1)
                    index = tuple(zip(index[0], index[1]))
                    var = stack([self.inav[ind] for ind in index])
                    v = var.map(square, inplace=False).nanmean().nanmean(axis=-2)
                    center = var.nanmean(axis=-2)
                    center.map(square)
                    center = center.nanmean()
                    int_vs_k.append((v - center) / center)
            if version is 'omega':
                int_vs_k = []
                for i, th in enumerate(thickness):
                    index = np.where(filt.transpose() == i+1)
                    index = tuple(zip(index[0], index[1]))
                    var = stack([self.inav[ind] for ind in index])
                    v = var.map(square, inplace=False).nanmean(axis=-2)
                    center = var.nanmean(axis=-2)
                    center.map(square)
                    center = center.nanmean()
                    int_vs_k.append(((v - center) / center).nanmean())
            int_vs_k = stack(int_vs_k)
            int_vs_k.axes_manager.navigation_axes[0].offset = thickness[0]
            int_vs_k.axes_manager.navigation_axes[0].scale = thickness[1] - thickness[0]
"""




class LazyPolarSignal(LazySignal,PolarSignal):

    _lazy = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)