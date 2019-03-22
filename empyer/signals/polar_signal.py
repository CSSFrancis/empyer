from empyer.signals.em_signal import EM_Signal
from empyer.signals.correlation_signal import CorrelationSignal
from empyer.misc.angular_correlation import angular_correlation


class PolarSignal(EM_Signal):
    _signal_type = "polar_signal"

    def __init__(self, *args, **kwargs):
        EM_Signal.__init__(self, *args, **kwargs)
        self.metadata.set_item("Signal.type", "polar_signal")

    def autocorrelation(self, binning_factor=1, cut=0, normalize=True, parallel=False):
        correlation = self.map(angular_correlation,
                               mask=self.get_mask(),
                               binning=binning_factor,
                               cut_off=cut,
                               normalize=normalize,
                               inplace=False)
        passed_meta_data = self.metadata.as_dictionary()
        if self.metadata.has_item('Masks'):
            del (passed_meta_data['Masks'])
        angular = CorrelationSignal(correlation, metadata=passed_meta_data)
        shift = cut // binning_factor
        angular.set_axes(0,
                         name=self.axes_manager[0].name,
                         scale=self.axes_manager[0].scale,
                         units=self.axes_manager[0].units)
        angular.set_axes(1,
                         name=self.axes_manager[1].name,
                         scale=self.axes_manager[1].scale,
                         units=self.axes_manager[1].units)
        angular.set_axes(2,
                         name="Radians",
                         scale=self.axes_manager[2].scale*binning_factor,
                         units="rad")
        offset = shift * self.axes_manager[3].scale*binning_factor
        angular.set_axes(3,
                         name="k",
                         scale=self.axes_manager[3].scale*binning_factor,
                         units=self.axes_manager[3].units,
                         offset=offset)
        return angular

