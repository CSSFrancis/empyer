from empyer.signals.em_signal import EM_Signal
from empyer.signals.power_signal import PowerSignal
from empyer.misc.angular_correlation import power_spectrum


class CorrelationSignal(EM_Signal):
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
        EM_Signal.__init__(self, *args, **kwargs)

    def get_power_spectrum(self, method="FFT"):
        """
        Calculate a power spectrum from the correlation signal
        Parameters
        ----------
        Method: String
            FFT- Fourier Transform
        """
        power_signal = self.map(power_spectrum,
                                method=method,
                                inplace=False)
        passed_meta_data = self.metadata.as_dictionary()
        if self.metadata.has_item('Masks'):
            del (passed_meta_data['Masks'])
        power = PowerSignal(power_signal)
        power.set_axes(0,
                       name=self.axes_manager[0].name,
                       scale=self.axes_manager[0].scale,
                       units=self.axes_manager[0].units)
        power.set_axes(1,
                       name=self.axes_manager[1].name,
                       scale=self.axes_manager[1].scale,
                       units=self.axes_manager[1].units)
        power.set_axes(2,
                       name="FourierCoefficient",
                       scale=1,
                       units="a.u.",
                       offset=2.5)
        power.set_axes(3,
                       name="k",
                       scale=self.axes_manager[3].scale,
                       units=self.axes_manager[3].units,
                       offset=self.axes_manager[3].offset)
        return power


