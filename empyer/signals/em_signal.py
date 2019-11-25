import numpy as np

from hyperspy.signals import Signal2D
from hyperspy.misc.slicing import SpecialSlicers
from hyperspy._signals.lazy import LazySignal


class EMSignal(Signal2D):
    """
    The Diffraction Signal class extends the Hyperspy 2d signal class
    """
    _signal_type = "em_signal"

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
        self.manav = MaskSlicer(self, isNavigation=True)
        self.masig = MaskSlicer(self, isNavigation=False)
        self.mask_passer = None

    def as_lazy(self, *args, **kwargs):
        res = super().as_lazy(*args, **kwargs)
        res.__class__ = LazyEMSignal
        res.__init__(**res._to_dictionary())
        return res

    def add_haadf_intensities(self, intensity_array, slope, intercept):
        """Add High Angle Annular Dark Field intensities for each point.

        Parameters
        -----------
        intensity_array: nd array
            An intensity array which is the same size of the navigation axis.  Acts as a measure of the thickness if
            there is a calculated normalized intensity. For masking in real space.  Matches data input NOT the real
            space coordinates from Hyperspy. (To match Hyperspy use np.transpose on intensity array)
        """
        if not self.metadata.has_item('HAADF'):
            self.metadata.add_node('HAADF.intensity')
            self.metadata.add_node('HAADF.filter_slope')
            self.metadata.add_node('HAADF.filter_intercept')
        if self.axes_manager.navigation_shape != np.shape(np.transpose(intensity_array)):
            print("The navigation axes and intensity array don't match")
            return
        self.metadata.HAADF.intensity = np.transpose(intensity_array)
        self.metadata.HAADF.filter_slope = slope
        self.metadata.HAADF.filter_intercept = intercept
        return

    def get_thicknesses(self):
        return self.metadata.HAADF.intensity*self.metadata.HAADF.filter_slope+self.metadata.HAADF.filter_intercept

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

    def mask_below(self, value, unmask=False):
        """Applies a mask to every pixel with an average value below value

        Parameters
        ----------
        value: float
            The maximum pixel value to apply a mask to.
        unmask: bool
            Unmask any pixel with a value below value
        """
        self.add_mask()
        self.data.mask[self.data < value] = not unmask

    def mask_above(self, value, unmask=False):
        """Applies a mask to every pixel with a value below some value

        Parameters
        ----------
        value: float
            The minimum pixel value to apply a mask to.
        unmask: bool
            Unmask any pixel with a value above value
        """
        self.add_mask()
        self.data.mask[self.data > value] = not unmask

    def mask_where(self, condition):
        """Mask at some condition

        Parameters:
        -------------
        condition: array_like
            Masking condition

        """
        self.add_mask()
        self.data.mask[condition] = True
        return

    def mask_border(self, pixels=1):
        self.add_mask()
        if not isinstance(pixels, int):
            pixels = (self.axes_manager.signal_axes[0].value2index(pixels))
        self.data.mask[..., -pixels:] = True
        self.data.mask[..., : pixels] = True
        self.data.mask[..., : pixels, :] = True
        self.data.mask[..., -pixels:, :] = True

    def add_mask(self):
        if not isinstance(self.data, np.ma.masked_array):
            self.data = np.ma.asarray(self.data)
            self.data.mask = False  # setting all values to unmasked

    def reset_mask(self):
        if isinstance(self.data, np.ma.masked_array):
            self.data.mask = False  # setting all values to unmasked

    def mask_circle(self, center, radius, unmask=False):
        # TODO: Add more shapes
        """Applies a mask to every pixel using a shape and the appropriate definition

        Parameters
        ----------
        shape: str
            Acceptable shapes ['rectangle, 'circle']
        data: list
            Define shapes. eg 'rectangle' -> [x1,x2,y1,y2] 'circle' -> [radius, x,y]
            data allows indexing with floats and the axes described for the signal
        unmask: bool
            Unmask any pixels in the defined shape
        """
        self.add_mask()
        if not all(isinstance(item, int) for item in center):
            center = (self.axes_manager.signal_axes[1].value2index(center[1]),
                     self.axes_manager.signal_axes[0].value2index(center[0]))
        if not isinstance(radius, int):
            radius = self.axes_manager.signal_axes[0].value2index(radius)
        x_ind, y_ind = np.meshgrid(range(-radius, radius + 1), range(-radius, radius + 1))
        r = np.sqrt(x_ind ** 2 + y_ind ** 2)
        inside = r < radius
        x_ind, y_ind = x_ind[inside]+int(center[0]), y_ind[inside]+int(center[1])
        self.data.mask[..., x_ind, y_ind] = True
        return

    def get_signal_axes_values(self):
        """ Returns the values for each pixel of the signal.  Useful for plotting without using hyperspy

        Returns
        ----------
        axis0: array-like
            The values for axis 0
        axis1: array-like
            The values for axis 1
        """
        axis0 = np.linspace(start=self.axes_manager.signal_axes[0].offset,
                            stop=(self.axes_manager.signal_axes[0].size *
                                  self.axes_manager.signal_axes[1].scale +
                                  self.axes_manager.signal_axes[0].offset),
                            num=self.axes_manager.signal_axes[0].size)
        axis1 = np.linspace(start=self.axes_manager.signal_axes[1].offset,
                            stop=(self.axes_manager.signal_axes[1].size *
                                  self.axes_manager.signal_axes[1].scale +
                                  self.axes_manager.signal_axes[1].offset),
                            num=self.axes_manager.signal_axes[1].size)
        return axis0, axis1


class MaskSlicer(SpecialSlicers):
    """
    Expansion of the Special Slicer class. Used for applying a mask
    """
    def __setitem__(self, key, value):
        if isinstance(self.obj, MaskPasser):
            array_slices = self.obj.signal._get_array_slices(key, self.isNavigation)
            if self.isNavigation == self.obj.isNavigation:
                print("You can't used masig or manav twice")
            self.obj.signal.add_mask()
            array_slices = tuple([slice1 if not (slice1 == slice(None, None, None)) else slice2 for
                                  slice1, slice2 in zip(self.obj.slice, array_slices)])
            self.obj.signal.data.mask[array_slices] = value
        else:
            array_slices = self.obj._get_array_slices(key, self.isNavigation)
            self.obj.add_mask()
            self.obj.data.mask[array_slices] = value

    def __getitem__(self, key, out=None):
        if isinstance(self.obj, MaskPasser):
            if self.isNavigation == self.obj.isNavigation:
                print("You can't used masig or manav twice")
                return
            array_slices = self.obj.signal._get_array_slices(key, self.isNavigation)
            array_slices = tuple([slice1 if not (slice1 == slice(None, None, None)) else slice2 for
                                  slice1, slice2 in zip(self.obj.slice, array_slices)])
            return MaskPasser(self.obj.signal, array_slices, self.isNavigation)
        else:
            array_slices = self.obj._get_array_slices(key, self.isNavigation)
            return MaskPasser(self.obj, array_slices, self.isNavigation)


class MaskPasser():
    def __init__(self, s, sl, nav):
        self.signal = s
        self.slice = sl
        self.isNavigation = nav
        self.manav = MaskSlicer(self, isNavigation=True)
        self.masig = MaskSlicer(self, isNavigation=False)

    def mask_below(self, maximum):
        """Mask below the max value

        Parameters:
        -------------
        maximum: float
            Mask any values in the slice below the maximum value
        """
        self.signal.add_mask()
        self.signal.data.mask[self.slice][(self.signal.data[self.slice] < maximum)] = True
        return

    def mask_above(self, minimum):
        """Mask above the minimum value

        Parameters:
        -------------
        minimum: float
            Mask any values in the slice above the minimum value
        """
        self.signal.add_mask()
        self.signal.data.mask[self.slice][(self.signal.data[self.slice] > minimum)] = True
        return

    def mask_where(self, condition):
        """Mask at some condition

        Parameters:
        -------------
        condition: array_like
            Masking condition

        """
        self.signal.add_mask()
        self.signal.data.mask[self.slice][condition] = True
        return

    def mask_circle(self, center, radius, unmask=False):
        # TODO: Add more shapes
        """Applies a mask to every pixel using a shape and the appropriate definition

        Parameters
        ----------
        center: tuple
            The (x,y) center of the circle
        radius: float or int
            The radius of the circle
        unmask: bool
            Unmask any pixels in the defined shape
        """
        self.signal.add_mask()
        if not all(isinstance(item, int) for item in center):
            center = (self.signal.axes_manager.signal_axes[1].value2index(center[1]),
                     self.signal.axes_manager.signal_axes[0].value2index(center[0]))
        if not isinstance(radius, int):
            radius = self.signal.axes_manager.signal_axes[0].value2index(radius)
        x_ind, y_ind = np.meshgrid(range(-radius, radius + 1), range(-radius, radius + 1))
        r = np.sqrt(x_ind ** 2 + y_ind ** 2)
        inside = r < radius
        x_ind, y_ind = x_ind[inside]+int(center[0]), y_ind[inside]+int(center[1])
        self.signal.data.mask[self.slice][..., x_ind, y_ind] = not unmask
        return


class LazyEMSignal(LazySignal,EMSignal):

    _lazy = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
