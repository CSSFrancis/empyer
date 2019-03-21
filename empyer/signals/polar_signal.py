from hyperspy.signals import Signal2D


class PolarSignal(Signal2D):
    _signal_type = "polar_signal"

    def __init__(self, *args, **kwargs):
        Signal2D.__init__(self, *args, **kwargs)

    def calculate_angular_correlation_spectrum(self, binning_factor=2, cut=45, normalize=True):
        images = self.data
        # getting dimensions for unwrapping
        signal_shapes = [axis.size for axis in self.axes_manager.signal_axes]
        navigation_shapes = [axis.size for axis in self.axes_manager.navigation_axes]
        unwrapped_length = [np.prod(navigation_shapes)]
        # unwrapping to allow dimension scaling
        images = np.reshape(images, (unwrapped_length + signal_shapes[::-1]))
        angular_array = np.array([angular_correlation(image, binning=binning_factor, cut=cut,
                                                      mask_below=self.metadata.Signal.mask, normalize=normalize)
                                  for image in images])
        shift = cut // binning_factor
        signal_shapes = list(np.floor_divide(signal_shapes, binning_factor))
        signal_shapes[1] = signal_shapes[1] - shift
        angular_array = np.reshape(angular_array, (navigation_shapes[::-1] + signal_shapes[::-1]))
        passed_meta_data = self.metadata.as_dictionary()
        angular = AngularCorrelationSpectrums(angular_array, binning_factor=binning_factor, cut_off=cut,
                                              metadata=passed_meta_data)
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