import numpy as np

from empyer.misc.ellipse_analysis import solve_ellipse
from empyer.misc.cartesain_to_polar import convert
from empyer.signals.em_signal import EM_Signal
from empyer.signals.polar_signal import PolarSignal


class DiffractionSignal(EM_Signal):
    """
    The Diffraction Signal class extends the Hyperspy 2d signal class
    """
    _signal_type = "diffraction_signal"

    def __init__(self, *args, **kwargs):
        """Create a Diffraction Signal from a numpy array.

        Parameters
        -------
        data : numpy array
           The signal data. It can be an array of any dimensions.
        axes : dictionary, optional
            Dictionary to define the axes (see the
            documentation of the AxesManager class for more details).
        attributes : dictionary, optional
            A dictionary whose items are stored as attributes.
        metadata : dictionary, optional
            A dictionary containing a set of parameters
            that will to stores in the `metadata` attribute.
            Some parameters might be mandatory in some cases.
        original_metadata : dictionary, optional
            A dictionary containing a set of parameters
            that will to stores in the `original_metadata` attribute. It
            typically contains all the parameters that has been
            imported from the original data file.
        """
        EM_Signal.__init__(self, *args, **kwargs)
        if not hasattr(self.metadata.Signal, 'Ellipticity.calibrated'):
            self.metadata.set_item("Signal.Ellipticity.calibrated", False)

        if not hasattr(self.metadata.Signal, 'type'):
            self.metadata.set_item("Signal.type", "diffraction_signal")

    def determine_ellipse(self, num_points=500, interactive=False, plot=False):
        # TODO: Identify if there needs to be a drift correction applied.
        # TODO: Exclude the zero beam if it isn't masked.
        """Determine the elliptical nature of the diffraction pattern.

        Parameters
        -------
        interactive : Boolean
            'interactive' nature means that points are chosen to create a ring
        axis : int
            'axis' to determine ellipse along
        num_points : int
            number of points to define ellipse by (only used if interactive = False)
        plot : Boolean
            Weather or not to plot the ellipse

        Returns
        -------
        center : list of int
            the center of the ellipse
        lengths : list of int
            the length in pixels of the major and minor axes
        angle : float
            the angle of the major axes
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
        """Take the Diffraction Pattern and unwrap the diffraction pattern.

        Parameters
        -------
        phase_width: int
            The number of pixels in the x direction
        radius: int
            The number of pixels in the y direction
        parallel: boolean
            use multiple processors for calculations (useful for large numbers of diffraction patterns,
            more for large pixel size)
        inplace: boolean
            replaces diffraction pattern data with polar equivalent

        Returns
        -------
        polar: PolarSignal
            Polar signal returned
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
        if self.metadata.has_item('Mask'):
            new_mask = convert(self.get_mask(),
                               center=self.metadata.Signal.Ellipticity.center,
                               angle=self.metadata.Signal.Ellipticity.angle,
                               foci=self.metadata.Signal.Ellipticity.lengths,
                               phase_width=phase_width,
                               radius=radius)
            new_mask = new_mask > 0
            new_mask = np.transpose(new_mask)
        else:
            new_mask = None

        passed_meta_data = self.metadata.as_dictionary()
        if self.metadata.Signal.has_item('Ellipticity'):
            del(passed_meta_data['Signal']['Ellipticity'])
        if self.metadata.has_item('Mask'):
            del (passed_meta_data['Mask'])

        polar = PolarSignal(polar_signal, metadata=passed_meta_data)
        polar.set_mask(mask=new_mask)

        polar.axes_manager.navigation_axes = self.axes_manager.navigation_axes
        polar.set_axes(-2,
                       name="Radians",
                       scale=2*np.pi/phase_width,
                       units="rad")
        polar.set_axes(-1,
                       name="k",
                       scale=self.axes_manager[-1].scale,
                       units=self.axes_manager[-1].units)
        return polar
