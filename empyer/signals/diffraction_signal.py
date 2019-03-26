from hyperspy.signals import Signal2D
from empyer.misc.ellipse_analysis import solve_ellipse
from empyer.misc.cartesain_to_polar import convert
import numpy as np


class DiffractionSignal(Signal2D):
    """
    The Diffraction Signal class extends the Hyperspy 2d signal class
    """
    _signal_type = "diffraction_signal"

    def __init__(self, *args, **kwargs):
        Signal2D.__init__(self, *args, **kwargs)
        if not hasattr(self.metadata.Signal, 'Ellipticity.calibrated'):
            self.metadata.set_item("Signal.Ellipticity.calibrated", False)

        if not hasattr(self.metadata.Signal, 'type'):
            self.metadata.set_item("Signal.type", "diffraction_signal")

    def get_mask(self):
        if self.metadata.has_item('Masks'):
            mask = np.zeros(shape=self.axes_manager.signal_shape, dtype=bool)
            mask_dict = self.metadata.Masks.as_dictionary()
            print(mask_dict)
            for key in mask_dict:
                m = mask_dict[key]
                if m['type'] is 'rectangle':
                    mask[int(m['data'][0]):int(m['data'][1]),int(m['data'][2]):int(m['data'][3])] = True
        else:
            return None
        return mask

    def add_mask(self, name='mask', type='rectangle', data=[1,1,1,1]):
        data_shape = np.shape(data)
        if len(data_shape) == 1:
            pass
        else:
            raise ValueError("Navigation shape of the marker must be 1 or the same navigation shape as this signal.")
        if not self.metadata.has_item('Masks'):
            self.metadata.add_node('Masks')
            self.metadata.Masks = {}
        self.metadata.Masks[name]= {'type':type,'data':data}

    def determine_ellipse(self, axis=None, num_points=500, interactive=False, plot=False):
        """
        Determine the elliptical nature of the diffraction pattern.
        # ToDo: Determine if the elliptical nature changes from pattern to pattern.
        :param interactive: Interactive nature means that points are chosen to create a ring
        :param axis: axis to determine ellipse along
        :param num_points: number of points to define ellipse by (only used if interactive = False)
        :param plot:
        :return: center: the center of the ellipse
        (or list of dim(self.sum(axis))
        :return: lengths: the length in pixels of the major and minor axes
        (or list of  dim(self.sum(axis))
        :return: angle: the angle of the major axes
        (or list of  dim(self.sum(axis))
        """
        center, lengths, angle = solve_ellipse(self.sum(axis=axis),
                                               axis=axis,
                                               mask=self.get_mask(),
                                               num_points=num_points,
                                               interactive=interactive,
                                               plot=plot)
        self.metadata.set_item("Signal.Ellipticity.center", center)
        self.metadata.set_item("Signal.Ellipticity.angle", angle)
        self.metadata.set_item("Signal.Ellipticity.lengths", lengths)
        self.metadata.set_item("Signal.Ellipticity.calibrated", True)
        self.metadata.set_item("Signal.Ellipticity.axis", axis)

        return center, lengths, angle

    def calculate_polar_spectrum(self, phase_width=720, radius=180):
        """
        Take the Diffraction Pattern and unwrap the diffraction pattern.
        :param phase_width: The number of pixels in the x direction
        :param radius: The number of pixels in the y direction
        :return:
        """

        if not self.metadata.Signal.Ellipticity.calibrated:
            self.determine_ellipticity()
        if self.metadata.Signal.Ellipticity.axis is None:
            images = [self.data]
        else:

            same_ellipticity_axis = np.subtract(-2,self.metadata.Signal.Ellipticity.axis)
            dif_ellipticity_axis =
            img_shape  = np.shape(self.data)
            images = np.reshape(self.data,(,*img_shape[-2:]))
        for i, img in enumerate(images):
            convert(img,
                    self.metadata.Signal.Ellipticity.center[i],
                    self.metadata.Signal.Ellipticity.angle[i],
                    self.metadata.Signal.Ellipticity.lengths[i])
        # getting dimensions for unwrapping
        signal_shapes = [axis.size for axis in self.axes_manager.signal_axes]
        navigation_shapes = [axis.size for axis in self.axes_manager.navigation_axes]
        unwrapped_length = [np.prod(navigation_shapes)]
        # unwrapping to allow dimension scaling
        images = np.reshape(images, (unwrapped_length+signal_shapes))
        polar_array = np.array([convert(image, self.metadata.Signal.Ellipticity.center,
                                        self.metadata.Signal.Ellipticity.angle,
                                        self.metadata.Signal.Ellipticity.lengths, phase_width=phase_width,
                                        radius=radius, plot=False)for image in images])
        polar_array = np.reshape(polar_array, (navigation_shapes[::-1] + [radius, phase_width]))
        passed_meta_data = self.metadata.as_dictionary()
        del(passed_meta_data['Signal']['Ellipticity'])
        polar = PolarSpectrums(polar_array, metadata=passed_meta_data)
        polar.set_axes(0,
                       name=self.axes_manager[0].name,
                       scale=self.axes_manager[0].scale,
                       units=self.axes_manager[0].units)
        polar.set_axes(1,
                       name=self.axes_manager[1].name,
                       scale=self.axes_manager[1].scale,
                       units=self.axes_manager[1].units)
        polar.set_axes(2,
                       name="Radians",
                       scale=2*np.pi/phase_width,
                       units="rad")
        polar.set_axes(3,
                       name="k",
                       scale=self.axes_manager[3].scale,
                       units=self.axes_manager[3].units)
        return polar