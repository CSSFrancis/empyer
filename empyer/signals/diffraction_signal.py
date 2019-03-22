import numpy as np

from empyer.misc.ellipse_analysis import solve_ellipse
from empyer.misc.cartesain_to_polar import convert
from empyer.signals.em_signal import EM_Signal
from empyer.signals.polar_signal import PolarSignal


class DiffractionSignal(EM_Signal):
    """
    The Diffraction Signal class extends the Spectrums class and the Hyperspy 2d signal class
    """
    _signal_type = "diffraction_signal"

    def __init__(self, *args, **kwargs):
        EM_Signal.__init__(self, *args, **kwargs)
        if not hasattr(self.metadata.Signal, 'Ellipticity.calibrated'):
            self.metadata.set_item("Signal.Ellipticity.calibrated", False)

        if not hasattr(self.metadata.Signal, 'type'):
            self.metadata.set_item("Signal.type", "diffraction_signal")

    def determine_ellipse(self, num_points=500, interactive=False, plot=False):
        """
        Determine the elliptical nature of the diffraction pattern.
        # ToDo: Determine if the elliptical nature changes from pattern to pattern.
        # ToDo: Allow for multiple ellipses to be associated with one diffraction pattern
        :param interactive: Interactive nature means that points are chosen to create a ring
        :param axis: axis to determine ellipse along
        :param num_points: number of points to define ellipse by (only used if interactive = False)
        :param plot:
        :return: center: the center of the ellipse
        :return: lengths: the length in pixels of the major and minor axes
        :return: angle: the angle of the major axes

        """
        center, lengths, angle = solve_ellipse(self.sum().data,
                                               mask=self.get_mask(),
                                               num_points=num_points,
                                               interactive=interactive,
                                               plot=plot)
        self.metadata.set_item("Signal.Ellipticity.center", center)
        self.metadata.set_item("Signal.Ellipticity.angle", angle)
        self.metadata.set_item("Signal.Ellipticity.lengths", lengths)
        self.metadata.set_item("Signal.Ellipticity.calibrated", True)
        return center, lengths, angle

    def calculate_polar_spectrum(self, phase_width=720, radius=None, parallel=False, inplace=False):
        """
        Take the Diffraction Pattern and unwrap the diffraction pattern.
        #ToDo Create a method which creates a new polar mask to be passed.
        :param phase_width: The number of pixels in the x direction
        :param radius: The number of pixels in the y direction
        :param parallel: use multiple processors for calculations (useful for large numbers of diffraction patterns,
         more for large pixel size)
        :param inplace: replaces diffraction pattern data with polar equivalent
        :return:
        """

        if not self.metadata.Signal.Ellipticity.calibrated:
            self.determine_ellipse()
        polar_signal = self.map(convert,
                                mask=self.get_mask(),
                                center=self.metadata.Signal.Ellipticity.center,
                                angle=self.metadata.Signal.Ellipticity.angle,
                                foci=self.metadata.Signal.Ellipticity.lengths,
                                phase_width=phase_width,
                                radius=radius,
                                parallel=parallel,
                                inplace=inplace)

        passed_meta_data = self.metadata.as_dictionary()
        if self.metadata.Signal.has_item('Ellipticity'):
            del(passed_meta_data['Signal']['Ellipticity'])
        if self.metadata.has_item('Masks'):
            del (passed_meta_data['Masks'])

        polar = PolarSignal(polar_signal, metadata=passed_meta_data)
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
