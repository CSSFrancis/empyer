import numpy as np

from empyer.signals.em_signal import EM_Signal


class PowerSignal(EM_Signal):
    _signal_type = "diffraction_signal"

    def __init__(self, *args, **kwargs):
        EM_Signal.__init__(self, *args, **kwargs)

    def simplify_symmetry(self, scale=1):
        scaled_power = [spectra > scale for spectra in self.data]
        return scaled_power

    def get_i_vs_k(self, normalize=True):
        images = self.data
        signal_shapes = [axis.size for axis in self.axes_manager.signal_axes]
        navigation_shapes = [axis.size for axis in self.axes_manager.navigation_axes]
        unwrapped_length = [np.prod(navigation_shapes)]
        images = np.reshape(images, (unwrapped_length + signal_shapes[::-1]))
        if normalize:
            i_vs_k = [np.divide(np.subtract(i, i.min()), (i.max()-i.min()))for i in images]
        print(np.shape(i_vs_k))
        i_vs_k = np.sum(i_vs_k, axis=0)
        _, k = self.get_signal_axes_values()
        return k, i_vs_k

    def get_map(self, k_region=[3, 6], normalize=True):
        images = self.data
        if normalize:
            images = [np.divide(np.subtract(i, i.min()), (i.max() - i.min())) for i in images]
        signal_shapes = [axis.size for axis in self.axes_manager.signal_axes]
        navigation_shapes = [axis.size for axis in self.axes_manager.navigation_axes]
        unwrapped_length = [np.prod(navigation_shapes)]
        images = np.reshape(images, (unwrapped_length + signal_shapes[::-1]))
        _, k = self.get_signal_axes_values()
        k = np.multiply(k > k_region[0], k < k_region[1])
        sliced_images = images[:, k, :]
        symmetry_map = np.sum(sliced_images, axis=1)
        symmetry_map = np.reshape(symmetry_map, (navigation_shapes[::-1] + [self.axes_manager.signal_axes[0].size]))
        return symmetry_map