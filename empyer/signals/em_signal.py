import numpy as np
import matplotlib.pyplot as plt

from hyperspy.signals import Signal2D
from hyperspy.misc.slicing import SpecialSlicers


class EM_Signal(Signal2D):
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
        self.mask_passer= None

    def add_hdaaf_intensities(self, intensity_array, slope, intercept):
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
        :return:
        """
        thickness = self.get_thicknesses()
        twosigma = 2 * np.std(thickness)
        deviation = np.abs(np.subtract(thickness,np.mean(thickness)))
        outside = deviation > twosigma
        return outside, np.mean(thickness), deviation

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
        if not isinstance(self.data, np.ma.masked_array):
            self.data = np.ma.asarray(self.data)
            self.data.mask = np.zeros(shape=self.data.shape, dtype=bool)
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
        if not isinstance(self.data, np.ma.masked_array):
            self.data = np.ma.asarray(self.data)
            self.data.mask = np.zeros(shape=self.data.shape, dtype=bool)
        self.data.mask[self.data > value] = not unmask

    def mask_shape(self, shape='rectangle', data=[1, 1, 1, 1], unmask=False):
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
        if shape is 'rectangle':
            self.mask_slice(data[0], data[1], data[2], data[3], unmask=unmask)
            return
        if not self.metadata.has_item('Mask'):
            self.metadata.add_node('Mask')
            mask = np.zeros(shape=tuple(reversed(self.axes_manager.signal_shape)), dtype=bool)
            self.metadata.Mask = mask
        if shape is 'circle':
            if not all(isinstance(item, int) for item in data):
                radius = self.axes_manager.signal_axes[0].value2index(data[0])
                y = self.axes_manager.signal_axes[0].value2index(data[1])
                x = self.axes_manager.signal_axes[1].value2index(data[2])
            else:
                radius = data[0]
                y = data[1]
                x = data[2]
            x_ind, y_ind = np.meshgrid(range(-radius, radius + 1), range(-radius, radius + 1))
            r = np.sqrt(x_ind ** 2 + y_ind ** 2)
            inside = r < radius
            x_ind, y_ind = x_ind[inside]+int(x), y_ind[inside]+int(y)
            self.metadata.Mask[x_ind, y_ind] = True
        return

    def mask_slice(self, x1, x2, y1, y2, unmask=False):
        """Applies a mask to some slice of the data (same as mask_shape for shape= 'rectangle')

            Does not support inav and isig if they are changed...
        Parameters
        ----------
        x1: float or int
        x2: float or int
        y1: float or int
        y2: float or int
        unmask: bool
            Unmask any pixels in the defined shape
        """

        if not isinstance(self.data, np.ma.masked_array):
            self.data = np.ma.asarray(self.data)
            self.data.mask = np.zeros(shape=self.data.shape, dtype=bool)

        if not all(isinstance(item, int) for item in [x1, x2, y1, y2]):
            x1 = self.axes_manager.signal_axes[0].value2index(x1)
            x2 = self.axes_manager.signal_axes[0].value2index(x2)
            y1 = self.axes_manager.signal_axes[1].value2index(y1)
            y2 = self.axes_manager.signal_axes[1].value2index(y2)

        if unmask is False:
            self.data.mask[..., y1:y2, x1:x2] = True
        if unmask is True:
            self.metadata.Mask[..., y1:y2, x1:x2] = False

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

    def get_mask(self):
        """ Returns the defined mask for the signal or None if no mask is defined

              Returns
              ----------
              mask: array-like
              """
        if self.metadata.has_item('Mask'):
            mask = self.metadata.Mask
        else:
            return None
        return mask

    def mean(self, axis=None, out=None, rechunk=True, mask=True):
        """Returns the signal average over an axis.

        Parameters
        ----------
        axis %s
        %s
        %s

        Returns
        -------
        s : Signal

        See also
        --------
        max, min, sum, std, var, indexmax, valuemax, amax

        Examples
        --------
        >>> import numpy as np
        >>> s = BaseSignal(np.random.random((64,64,1024)))
        >>> s.data.shape
        (64,64,1024)
        >>> s.mean(-1).data.shape
        (64,64)

        """
        if axis is None:
            axis = self.axes_manager.navigation_axes
        #if mask is True and self.metadata.has_item('Mask'):

        return self._apply_function_on_data_and_remove_axis(
            np.mean, axis, out=out, rechunk=rechunk)


class MaskSlicer(SpecialSlicers):

    def __setitem__(self, key, value):
        if isinstance(self.obj, MaskPasser):
            array_slices = self.obj.signal._get_array_slices(key, self.isNavigation)
            if self.isNavigation == self.obj.isNavigation:
                print("You can't used masig or manav twice")
            if not isinstance(self.obj.signal.data, np.ma.masked_array):
                self.obj.signal.data = np.ma.asarray(self.obj.signal.data)
                self.obj.signal.data.mask = False  # setting all values to unmasked
            self.obj.signal.data.mask[self.obj.slice][array_slices] = value
        else:
            array_slices = self.obj._get_array_slices(key, self.isNavigation)
            if not isinstance(self.obj.data, np.ma.masked_array):
                self.obj.data = np.ma.asarray(self.obj.data)
                self.obj.data.mask = False  # setting all values to unmasked
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

    def mask_below(self, max):
        if not isinstance(self.signal.data, np.ma.masked_array):
            self.signal.data = np.ma.asarray(self.signal.data)
            self.signal.data.mask = False  # setting all values to unmasked
            print(self.slice)
            print(self.signal.data[self.slice] < min)
        self.signal.data.mask[self.slice][(self.signal.data[self.slice] < max)] = True
        return

    def mask_above(self, min):
        if not isinstance(self.signal.data, np.ma.masked_array):
            self.signal.data = np.ma.asarray(self.obj.data)
            self.signal.data.mask = False  # setting all values to unmasked
        self.signal.data.mask[self.slice][(self.signal.data[self.slice] > min)] = True
        return

    def mask_where(self, condition):
        if not isinstance(self.signal.data, np.ma.masked_array):
            self.signal.data = np.ma.asarray(self.obj.data)
            self.signal.data.mask = False  # setting all values to unmasked
        self.signal.data.mask[self.slice][(condition)] = True
        return
