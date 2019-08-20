from empyer.signals.em_signal import EMSignal
from empyer.signals.power_signal import PowerSignal
from empyer.misc.angular_correlation import power_spectrum
from hyperspy._signals.lazy import LazySignal


class CorrelationSignal(EMSignal):
    """Create a  Correlation Signal from a numpy array.

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
    _signal_type = "correlation_signal"

    def __init__(self, *args, **kwargs):
        EMSignal.__init__(self, *args, **kwargs)
        self.metadata.set_item("Signal.type", "correlation_signal")

    def as_lazy(self, *args, **kwargs):
        res = super().as_lazy(*args, **kwargs)
        res.__class__ = LazyCorrelationSignal
        res.__init__(**res._to_dictionary())
        return res

    def get_power_spectrum(self, method="FFT"):
        """
        Calculate a power spectrum from the correlation signal

        Parameters
        ----------
        method : str
            'FFT' gives fourier transformation of the angular power spectrum.  Currently the only method available
        """
        power_signal = self.map(power_spectrum,
                                method=method,
                                inplace=False,
                                show_progressbar=False)
        passed_meta_data = self.metadata.as_dictionary()
        if self.metadata.has_item('Masks'):
            del (passed_meta_data['Masks'])
        power = PowerSignal(power_signal)
        power.axes_manager.navigation_axes = self.axes_manager.navigation_axes

        power.set_axes(-2,
                       name="FourierCoefficient",
                       scale=1,
                       units="a.u.",
                       offset=.5)
        power.set_axes(-1,
                       name="k",
                       scale=self.axes_manager[-1].scale,
                       units=self.axes_manager[-1].units,
                       offset=self.axes_manager[-1].offset)
        return power

    def get_summed_power_spectrum(self):
        # TODO: Add in the ability to get the summed power spectrum over an axis.
        summed_pow = self.sum(axis=(0, 1))
        return summed_pow.get_power_spectrum()


class LazyCorrelationSignal(LazySignal,CorrelationSignal):

    _lazy = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)