import numpy as np
from empyer.signals.emsignal import EMSignal
from empyer.signals.correlation_signal import CorrelationSignal
from empyer.misc.angular_correlation import angular_correlation
from empyer.misc.image import square


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
            cut = self.axes_manager.signal_axes[-3].value2index(cut)
        correlation = self.map(angular_correlation,
                               binning=binning_factor,
                               cut_off=cut,
                               normalize=normalize,
                               inplace=False)
        correlation.mask_where(condition=(correlation.data ==0))
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

    def fem(self, version="omega"):
        """Calculated the variance among some image

        Parameters
        ----------
        version : str
            The name of the FEM equation to use. 'rings' calculates the mean of the variances of all the patterns at
            some k.  'omega' calculates the variance of the annular means for every value of k.
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
            filter, thickness = self.thickness_filter()
            with self.unfolded(unfold_navigation=True, unfold_signal=False):
                if version is 'rings':
                    int_vs_k = []
                    for i, th in enumerate(thickness):
                        var = self.inav[filter == i].nanmean(axis=-1)
                        var.map(square)
                        var = var.nanmean()
                        center = self.nanmean(axis=-1).nanmean()
                        center.map(square)
                        int_vs_k = int_vs_k.append((var - center) / center)
                elif version is 'omega':
                    int_vs_k = []
                    for i, th in enumerate(thickness):
                        index = np.where(filter.flatten(0)==i+1)[0]
                        print(tuple(index))
                        var = self.inav[tuple(index)].map(square,
                                                         show_progressbar=False,
                                                         inplace=False).nanmean().nanmean(axis=1)
                        center = self.nanmean(axis=-1)
                        center.map(square)
                        center = center.nanmean()
                        int_vs_k.append((var - center) / center)
        return int_vs_k

