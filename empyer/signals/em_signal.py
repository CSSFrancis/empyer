import numpy as np
import matplotlib.pyplot as plt

from hyperspy.signals import Signal2D


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

    def set_mask(self, mask):
        """Set a mask

        Parameters
        ----------
        mask: n x m boolean array
            A boolean array which is set as the mask for some signal.  Will overwrite the mask. Function accepts m x n
            and n x m arrays and will take the transpose of the signal so that it matches the hyperspy signal axes.
        """
        if not self.metadata.has_item('Mask'):
            self.metadata.add_node('Mask')
        if tuple(reversed(self.axes_manager.signal_shape)) == np.shape(mask):
            self.metadata.Mask = mask
        elif self.axes_manager.signal_shape == np.shape(mask):
            self.metadata.Mask = np.transpose(mask)
        else:
            print('The mask shape must be the same shape as the signal')

    def show_mask(self):
        """Plots the mask in a separate window
        """
        # TODO:Show mask overlaid over the signal
        ax = self.axes_manager.signal_axes
        ax = [a.get_axis_dictionary() for a in reversed(ax)]
        Signal2D(data=self.get_mask(), axes=ax).plot()
        plt.show()
        return

    def mask_below(self, value, unmask=False):
        """Applies a mask to every pixel with an average value below value

        Parameters
        ----------
        value: float
            The maximum pixel value to apply a mask to.
        unmask: bool
            Unmask any pixel with a value below value
        """
        if not self.metadata.has_item('Mask'):
            self.metadata.add_node('Mask')
            mask = np.zeros(shape=tuple(reversed(self.axes_manager.signal_shape)), dtype=bool)
            self.metadata.Mask = mask
        if unmask is False:
            is_below = self.mean().data < value
            self.metadata.Mask[is_below] = True
        if unmask is True:
            is_below = self.mean().data < value
            self.metadata.Mask[is_below] = False

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

        Parameters
        ----------
        x1: float or int
        x2: float or int
        y1: float or int
        y2: float or int
        unmask: bool
            Unmask any pixels in the defined shape
        """
        if not self.metadata.has_item('Mask'):
            self.metadata.add_node('Mask')

            mask = np.zeros(shape=tuple(reversed(self.axes_manager.signal_shape)), dtype=bool)
            self.metadata.Mask = mask
        if not all(isinstance(item, int) for item in [x1, x2, y1, y2]):
            x1 = self.axes_manager.signal_axes[0].value2index(x1)
            x2 = self.axes_manager.signal_axes[0].value2index(x2)
            y1 = self.axes_manager.signal_axes[1].value2index(y1)
            y2 = self.axes_manager.signal_axes[1].value2index(y2)
        if unmask is False:
            self.metadata.Mask[y1:y2, x1:x2] = True
        if unmask is True:
            self.metadata.Mask[y1:y2, x1:x2] = False

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
