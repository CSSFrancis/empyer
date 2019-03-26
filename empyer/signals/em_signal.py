import numpy as np

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

    def get_signal_axes_values(self):
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
        if self.metadata.has_item('Masks'):
            mask = np.zeros(shape=self.axes_manager.signal_shape, dtype=bool)
            mask_dict = self.metadata.Masks.as_dictionary()
            for key in mask_dict:
                m = mask_dict[key]
                if m['type'] is 'rectangle':
                    mask[int(m['data'][0]):int(m['data'][1]),int(m['data'][2]):int(m['data'][3])] = True

        else:
            return None
        print(np.shape(mask))
        return np.transpose(mask)

    def add_mask(self, name='mask', type='rectangle', data=[1,1,1,1]):
        """Add a mask by using four points to define a rectangle

        Parameters
        ----------
        name : str
           the name of the mask (overwriting the name replaces the mask)
        type : str
            The shape of the mask (rectangle...)
        data : array_like
            Description of the shape
        """
        data_shape = np.shape(data)
        if len(data_shape) == 1:
            pass
        else:
            raise ValueError("Navigation shape of the marker must be 1 or the same navigation shape as this signal.")
        if not self.metadata.has_item('Masks'):
            self.metadata.add_node('Masks')
            self.metadata.Masks = {}
        self.metadata.Masks[name]={'type':type,'data':data}
