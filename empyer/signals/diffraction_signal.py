import numpy as np

from empyer.misc.ellipse_analysis import solve_ellipse
from empyer.misc.cartesain_to_polar import convert
from empyer.signals.em_signal import EMSignal
from empyer.signals.polar_signal import PolarSignal
from hyperspy._signals.lazy import LazySignal


class DiffractionSignal(EMSignal):
    """
    The Diffraction Signal class extends the Hyperspy 2d signal class
    """
    _signal_type = "DiffractionSignal"

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
        EMSignal.__init__(self, *args, **kwargs)
        if not hasattr(self.metadata.Signal, 'Ellipticity.calibrated'):
            self.metadata.set_item("Signal.Ellipticity.calibrated", False)

        if not hasattr(self.metadata.Signal, 'type'):
            self.metadata.set_item("Signal.type", "diffraction_signal")

    def as_lazy(self, *args, **kwargs):
        res = super().as_lazy(*args, **kwargs)
        res.__class__ = LazyDiffractionSignal
        res.__init__(**res._to_dictionary())
        return res

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
                                               num_points=num_points,
                                               interactive=interactive,
                                               plot=plot)
        self.metadata.set_item("Signal.Ellipticity.center", center)
        self.metadata.set_item("Signal.Ellipticity.angle", angle)
        self.metadata.set_item("Signal.Ellipticity.lengths", lengths)
        self.metadata.set_item("Signal.Ellipticity.calibrated", True)
        return center, lengths, angle

    def calculate_polar_spectrum(self,
                                 phase_width=720,
                                 radius=[0, -1],
                                 parallel=False,
                                 inplace=False,
                                 segments=None,
                                 num_points=500):
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
        rag = None
        if self._lazy:
            rag = False
        if radius[1] == -1:
            radius[1] = int(min(np.subtract(self.axes_manager.signal_shape, self.metadata.Signal.Ellipticity.center))-1)
        if segments is None:
            print(rag)
            polar_signal = self.map(convert,
                                    center=self.metadata.Signal.Ellipticity.center,
                                    angle=self.metadata.Signal.Ellipticity.angle,
                                    lengths=self.metadata.Signal.Ellipticity.lengths,
                                    phase_width=phase_width,
                                    radius=radius,
                                    parallel=parallel,
                                    inplace=inplace,
                                    show_progressbar=False)
            print(np.shape(polar_signal))
        else:
            len_of_segments = np.array(self.axes_manager.navigation_shape) // segments
            extra_len = np.array(self.axes_manager.navigation_shape) % segments
            centers = np.zeros(shape=(*self.axes_manager.navigation_shape, 2))
            lengths = np.zeros(shape=(*self.axes_manager.navigation_shape, 2))
            angle = np.zeros(shape=self.axes_manager.navigation_shape)
            for i in range(segments):
                for j in range(segments):
                    extra = [extra_len[0] * (i == segments-1), extra_len[1] * (j == segments-1)]
                    s1 = int(i*len_of_segments[0])
                    sp1 = int((i+1)*len_of_segments[0]+extra[0])
                    s2 = int(j*len_of_segments[1])
                    sp2 = int((j+1)*len_of_segments[1]+extra[1])
                    centers[s1:sp1, s2:sp2, :], lengths[s1:sp1, s2:sp2, :], angle[s1:sp1, s2:sp2] = solve_ellipse(self.inav[s1:sp1, s2:sp2].sum().data,num_points=num_points)

            ellip = (('center', np.reshape(centers, (-1, 2))),
                     ('lengths', np.reshape(lengths, (-1, 2))),
                     ('angle', np.reshape(angle, -1)))
            polar_signal = self._map_iterate(convert,
                                             iterating_kwargs=ellip,
                                             parallel=parallel,
                                             inplace=inplace,
                                             show_progressbar=False,
                                             radius=radius,
                                             phase_width=phase_width)
            print(polar_signal.inav[1, 1])

            print("done")


        passed_meta_data = self.metadata.as_dictionary()
        if self.metadata.Signal.has_item('Ellipticity'):
            del(passed_meta_data['Signal']['Ellipticity'])

        polar = PolarSignal(polar_signal, metadata=passed_meta_data)
        polar.mask_below(value=-9)

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


class LazyDiffractionSignal(LazySignal, DiffractionSignal):

    _lazy = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)