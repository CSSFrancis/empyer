from hyperspy.misc.utils import add_scalar_axis
import numpy as np

def map_result_construction(signal,
                            inplace,
                            result,
                            ragged,
                            sig_shape=None,
                            lazy=False,
                            is_navigation=False):
    from hyperspy.signal import BaseSignal
    from hyperspy._lazy_signals import LazySignal
    res = None
    print(result[1,1,1,1])
    if inplace:
        sig = signal
    else:
        res = sig = signal._deepcopy_with_new_data()
    if ragged:
        sig.data = result
        sig.axes_manager.remove(sig.axes_manager.signal_axes)
        sig.__class__ = LazySignal if lazy else BaseSignal
        sig.__init__(**sig._to_dictionary(add_models=True))

    else:
        if not sig._lazy and sig.data.shape == result.shape and np.can_cast(
                result.dtype, sig.data.dtype):
            sig.data[:] = result
            print("Here")
            print(sig.inav[1, 1].isig[1, 1:3].data)
        else:
            sig.data = result

        # remove if too many axes
        sig.axes_manager.remove(sig.axes_manager.signal_axes[len(sig_shape):])
        # add additional required axes
        if is_navigation:
            for ind in range(len(sig_shape) - sig.axes_manager.signal_dimension, 0, -1):
                sig.axes_manager._append_axis(sig_shape[-ind], navigate=True)
        else:
            for ind in range(len(sig_shape) - sig.axes_manager.signal_dimension, 0, -1):
                sig.axes_manager._append_axis(sig_shape[-ind], navigate=False)
    sig.get_dimensions_from_data()
    if not sig.axes_manager._axes:
        add_scalar_axis(sig)
    return res
