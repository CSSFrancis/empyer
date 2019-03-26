from hyperspy.api import load as hsload


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
