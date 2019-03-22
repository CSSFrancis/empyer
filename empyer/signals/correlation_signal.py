from empyer.signals.em_signal import EM_Signal
from empyer.signals.power_signal import PowerSignal
from empyer.misc.angular_correlation import power_spectrum

class CorrelationSignal(EM_Signal):
    """
    Angular Correlations
    """
    _signal_type = "correlation_signal"

    def __init__(self, *args, **kwargs):
        EM_Signal.__init__(self, *args, **kwargs)

    def get_power_spectrum(self, method="FFT"):
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


