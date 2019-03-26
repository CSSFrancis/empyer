from hyperspy.api import load as hsload
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
    return hsload(filenames=filenames,
                  signal_type=signal_type,
                  stack=stack,
                  stack_axis=stack_axis,
                  new_axis_name=new_axis_name,
                  lazy=lazy,
                  **kwds)


def to_diffraction_signal(signal=None):
    ax = signal.axes_manager.as_dictionary()
    ds = DiffractionSignal(signal,
                           metadata=signal.metadata.as_dictionary())
    ds.axes_manager.update_axes_attributes_from(ax)

    return ds


def to_polar_signal(signal=None):
    ps = PolarSignal(signal,
                     metadata=signal.metadata.as_dictionary(),
                     axes=signal.axes_manager.as_dictionary())
    return ps
