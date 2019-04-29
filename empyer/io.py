from hyperspy.api import load as hsload
from .signals.em_signal import EM_Signal
from .signals.diffraction_signal import DiffractionSignal
from .signals.polar_signal import PolarSignal
from .signals.correlation_signal import CorrelationSignal
from .signals.power_signal import PowerSignal


def load(filenames=None,
         signal_type=None,
         stack=False,
         stack_axis=None,
         new_axis_name='stack_element',
         lazy=False,
         **kwds):
    """Extends the hyperspy loading functionality with additional ability to load empyer signals.

    Parameters
    ---------------------
    filenames:list
        The list of file paths or single file to be loaded
    signal_type: str
        The type of signal to be loaded. Can be read from file or given
    stack: bool
        Stack creates a new axis and loads a list of files into one signal
    new_axis_name: str
        The name of the new axis created with stack
    lazy: bool
        Load the signal into memory or load chunks.  May effect performance, but useful for large datasets.

    Returns
    ---------------------
    signal:
        A signal of type signal_type
    """
    signal = hsload(filenames=filenames,
                    signal_type=signal_type,
                    stack=stack,
                    stack_axis=stack_axis,
                    new_axis_name=new_axis_name,
                    lazy=lazy,
                    **kwds)
    if signal.metadata.Signal.signal_type == 'diffraction_signal':
        signal = to_diffraction_signal(signal)
    if signal.metadata.Signal.signal_type == 'em_signal':
        signal = to_em_signal(signal)
    if signal.metadata.Signal.signal_type == 'polar_signal':
        signal = to_polar_signal(signal)
    if signal.metadata.Signal.signal_type == 'correlation_signal':
        signal = to_correlation_signal(signal)
    if signal.metadata.Signal.signal_type == 'power_signal':
        signal = to_power_signal(signal)
    if signal.metadata.Signal.has_item('signal_type'):
        print(signal.metadata.Signal.signal_type, " loaded!")
    return signal


def to_em_signal(signal=None):
    """Hyperspy signal to em_signal
    """
    ax = signal.axes_manager.as_dictionary()
    ax = [ax[key]for key in ax]
    ds = EM_Signal(signal, metadata=signal.metadata.as_dictionary(), axes=ax)
    return ds


def to_diffraction_signal(signal=None):
    """Hyperspy signal to diffraction_signal
    """
    ax = signal.axes_manager.as_dictionary()
    ax = [ax[key]for key in ax]
    ds = DiffractionSignal(signal, metadata=signal.metadata.as_dictionary(), axes=ax)
    return ds


def to_polar_signal(signal=None):
    """Hyperspy signal to polar_signal
     """
    ax = signal.axes_manager.as_dictionary()
    ax = [ax[key]for key in ax]
    ps = PolarSignal(signal,
                     metadata=signal.metadata.as_dictionary(),
                     axes=ax)
    return ps


def to_correlation_signal(signal=None):
    """Hyperspy signal to correlation_signal
     """
    ax = signal.axes_manager.as_dictionary()
    ax = [ax[key]for key in ax]
    cs = CorrelationSignal(signal,
                     metadata=signal.metadata.as_dictionary(),
                     axes=ax)
    return cs


def to_power_signal(signal=None):
    """Hyperspy signal to power_signal
     """
    ax = signal.axes_manager.as_dictionary()
    ax = [ax[key]for key in ax]
    ps = PowerSignal(signal,
                     metadata=signal.metadata.as_dictionary(),
                     axes=ax)
    return ps
