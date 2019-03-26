import numpy as np

from hyperspy.signals import Signal2D


class EM_Signal(Signal2D):
    """ Basic unit of data in the program. Spectrums can be any dimension of array but all of the data should be of the
    same type from the same area... (Maybe define this better later... You can use it kind of however you want though)
    Extends the Signal2D Class from hyperspy so there is that added functionality
    :param data: numpy array or list of data.
    :param mask_below: values below this value are masked for all calculations
    :param dimensions: An array which describes the dimensions (maybe should add in description of data and navigation?)
    :param units: array of strings which describe the units
    :param offsets: offsets for plotting and of spectrum
    :param date_collected: ...
    ... For more parameters see hyperspy's Signal2D Calss
    """

    def __init__(self, *args, **kwargs):
        Signal2D.__init__(self, *args, **kwargs)

    def set_axes(self, index, name=None, scale=None, units=None, offset=None):
        """
        Set the axes of an 2d spectrum
        :param index: The indexes
        :param name:
        :param scale:
        :param units:
        :param offset:
        :return:
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
        data_shape = np.shape(data)
        if len(data_shape) == 1:
            pass
        else:
            raise ValueError("Navigation shape of the marker must be 1 or the same navigation shape as this signal.")
        if not self.metadata.has_item('Masks'):
            self.metadata.add_node('Masks')
            self.metadata.Masks = {}
        self.metadata.Masks[name]={'type':type,'data':data}
