import numpy as np
import dask.array as da

from hyperspy._signals.signal2d import Signal2D
from hyperspy._signals.lazy import LazySignal
from empyer.misc.masks import Mask


class Amorphous2D(Signal2D):
    """
    The Diffraction Signal class extends the Hyperspy 2d signal class
    """
    _signal_type = "amorphous2d"

    def __init__(self, *args, **kwargs):
        """Basic unit of data in the program.

        Spectrums can be any dimension of array but all of the data should be of the
        same type from the same area... (Maybe define this better later... You can use it kind of however you want though)
        Extends the Signal2D Class from hyperspy so there is that added functionality

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
        Notes: For more parameters see hyperspy's Signal2D Class
        """
        Signal2D.__init__(self, *args, **kwargs)
        self.manav = Mask(self,is_navigation=True)
        self.masig = Mask(self,is_navigation=False)
        if not self.metadata.has_item('Mask'):
            self.metadata.add_node('Mask.sig_mask')
            self.metadata.add_node("Mask.nav_mask")
            self.metadata.Mask.sig_mask = np.zeros(shape=self.axes_manager.signal_shape, dtype=bool)
            self.metadata.Mask.nav_mask = np.zeros(shape=self.axes_manager.navigation_shape, dtype=bool)

        if not self.metadata.has_item('Sum'):
            self.metadata.add_node("Sum.sig_sum")
            self.metadata.add_node("Sum.nav_sum")
            self.metadata.Sum.sig_sum = self.sum(axis=self.axes_manager.signal_axes)
            self.metadata.Mask.nav_sum = self.sum(axis=self.axes_manager.navigation_axes)

    def add_haadf_intensities(self, intensity_array, slope=None, intercept=None):
        """Add High Angle Annular Dark Field intensities for each point.

        Parameters
        -----------
        intensity_array: nd array
            An intensity array which is the same size of the navigation axis.  Acts as a measure of the thickness if
            there is a calculated normalized intensity. For masking in real space.  Matches data input NOT the real
            space coordinates from Hyperspy. (To match Hyperspy use np.transpose on intensity array)
        slope: None or float
            The slope to measure thickness from HAADF intensities
        intercept: None or float
            THe intercept to measure thickness from HAADF intensities
        """
        if not self.metadata.has_item('HAADF'):
            self.metadata.add_node('HAADF.intensity')
            self.metadata.add_node('HAADF.filter_slope')
            self.metadata.add_node('HAADF.filter_intercept')
        if self.axes_manager.navigation_shape != np.shape(np.transpose(intensity_array)):
            print("The navigation axes and intensity array don't match")
            return
        ax = [a.get_axis_dictionary() for a in self.axes_manager.navigation_axes]
        self.metadata.HAADF.intensity = Signal2D(data=np.transpose(intensity_array), axes=ax)
        self.metadata.HAADF.filter_slope = slope
        self.metadata.HAADF.filter_intercept = intercept
        return

    def axis_map(self,axes, function, show_progressbar=None, parallel=None, inplace=True, ragged=None, **kwargs):
        """Applies a 2-D filter on either the navigation or the signal axis
        Parameters
        --------------
        axes: list
            The indexes of the axes to be operated on for the 2D transformation
        function:function
            Any function that can be applied to a 2D signal
        show_progressbar: (None or bool)
            If True, display a progress bar. If None, the default from the preferences settings is used.
        parallel:(None or bool)
            If True, perform computation in parallel using multiple cores. If None, the default from the preferences
            settings is used.
        inplace: bool
            if True (default), the data is replaced by the result. Otherwise a new Signal with the results is returned.
        ragged:(None or bool)
            Indicates if the results for each navigation pixel are of identical shape (and/or numpy arrays to begin with). If None, the appropriate choice is made while processing. Note: None is not allowed for Lazy signals!
        **kwargs (dict)
            All extra keyword arguments are passed to the provided function
        """
        pass

    def as_lazy(self, *args, **kwargs):
        """ Change the signal to lazy loading signal.  For large signals.
        """
        res = super().as_lazy(*args, **kwargs)
        res.__class__ = LazyAmorphousSignal
        res.__init__(**res._to_dictionary())
        return res

    def center_direct_beam(self, center):
        pass

    def estimate_affine(self, method):
        pass

    def get_thicknesses(self):
        """ Returns a Signal2D object with the thicknesses based on the High Angular annular detector and the
        applied calibration.
        """
        if self.metadata.HAADF.filter_slope and self.metadata.HAADF.filter_intercept:
            return self.metadata.HAADF.intensity*self.metadata.HAADF.filter_slope+self.metadata.HAADF.filter_intercept
        else:
            print("You need a slope and an intercept to get the thicknesses from the High Angle Annular Dark field "
                  "Image")
            return

    def get_direct_beam_pos(self, method):
        pass

    def set_axes(self, index, name=None, scale=None, units=None, offset=None):
        """Set axes of the signal

        Parameters
        ----------
        index: int
           The index of the axes
        name: str
            The name of the axis
        scale : float
            The scale fo the axis
        units : str
            The units of the axis
        offset : float
            The offset of the axes
        """
        if name is not None:
            self.axes_manager[index].name = name
        if scale is not None:
            self.axes_manager[index].scale = scale
        if units is not None:
            self.axes_manager[index].units = units
        if offset is not None:
            self.axes_manager[index].offset = offset

    def thickness_filter(self):
        """Filter based on HAADF intensities

        Returns
        ------------------
        th_filter: array-like
            Integers which are used to filter into different thicknesses. Basically used to bin the
            signal
        thicknesses: 1-d array
            The thicknesses for the signal at every integer.
        """
        thickness = self.get_thicknesses()
        twosigma = 2 * np.std(thickness)
        deviation = np.subtract(thickness, np.mean(thickness))
        th_filter = thickness
        th_filter[deviation > twosigma] = 0
        th_filter[deviation < twosigma] = 0
        th_filter[(-twosigma < deviation) & (deviation <= -twosigma/2)] = 1
        th_filter[(-twosigma / 2 < deviation) & (deviation <= 0)] = 2
        th_filter[(0 < deviation) & (deviation <= twosigma/2)] = 3
        th_filter[(twosigma / 2 < deviation) & (deviation <= twosigma)] = 4
        thickness = [np.mean(thickness) - 3*twosigma/2,
                     np.mean(thickness) - twosigma/2,
                     np.mean(thickness) +twosigma/2,
                     np.mean(thickness) + 3 * twosigma / 2]
        return th_filter, thickness

    def to_polar(self, affine):
        pass

    def to_correlation(self):
        pass

    def get_virtual_image(self, roi):
        pass


class LazyAmorphousSignal(LazySignal, Amorphous2D):
    _lazy = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
