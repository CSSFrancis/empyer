import numpy as np
import dask.array as da
import dask.delayed as dd

from hyperspy._signals.signal2d import Signal2D
from hyperspy.signal import BaseSignal
from hyperspy._signals.lazy import LazySignal
from empyer.misc.masks import Mask
from empyer.misc.ellipse_analysis import solve_ellipse
from empyer.misc.cartesain_to_polar import to_polar_image
from hyperspy.defaults_parser import preferences
from hyperspy.docstrings.signal import SHOW_PROGRESSBAR_ARG, PARALLEL_ARG
from hyperspy.external.progressbar import progressbar
from empyer.misc.utils import map_result_construction
from empyer.signals.polar_amorphous2d import PolarAmorphous2D
from itertools import product


class Amorphous2D(Signal2D):
    """
    The Diffraction Signal class extends the Hyperspy 2d signal class
    """
    _signal_type = "amorphous2d"

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
        self.manav = Mask(self, is_navigation=True)
        self.masig = Mask(self, is_navigation=False)
        if not self.metadata.has_item('Mask'):
            self.metadata.add_node('Mask.sig_mask')
            self.metadata.add_node("Mask.nav_mask")
            self.metadata.Mask.sig_mask = np.zeros(shape=self.axes_manager.signal_shape, dtype=bool)
            self.metadata.Mask.nav_mask = np.zeros(shape=self.axes_manager.navigation_shape, dtype=bool)

        if not self.metadata.has_item('Sum'):
            self.metadata.add_node("Sum.sig_sum")
            self.metadata.add_node("Sum.nav_sum")
            self.metadata.Sum.sig_sum = self.sum(axis=self.axes_manager.navigation_axes)
            self.metadata.Mask.nav_sum = self.sum(axis=self.axes_manager.signal_axes)

    def add_haadf_intensities(self, intensity_array, slope=None, intercept=None):
        """Add High Angle Annular Dark Field intensities for each point.

        Parameters
        -----------
        intensity_array: nd array
            An intensity array which is the same size of the navigation axis.  Acts as a measure of the thickness if
            there is a calculated normalized intensity. For masking in real space.  Matches data input NOT the real
            space coordinates from Hyperspy. (To match Hyperspy use np.transpose on intensity array)
        slope: None or float
            The slope to measure thickness from HAADF intensities
        intercept: None or float
            THe intercept to measure thickness from HAADF intensities
        """
        if not self.metadata.has_item('HAADF'):
            self.metadata.add_node('HAADF.intensity')
            self.metadata.add_node('HAADF.filter_slope')
            self.metadata.add_node('HAADF.filter_intercept')
        if self.axes_manager.navigation_shape != np.shape(np.transpose(intensity_array)):
            print("The navigation axes and intensity array don't match")
            return
        ax = [a.get_axis_dictionary() for a in self.axes_manager.navigation_axes]
        self.metadata.HAADF.intensity = Signal2D(data=np.transpose(intensity_array), axes=ax)
        self.metadata.HAADF.filter_slope = slope
        self.metadata.HAADF.filter_intercept = intercept
        return

    def axis_map(self, function,
                 show_progressbar=None, parallel=None, inplace=True, ragged=None, is_navigation=False, scale=None,
                 units=None, offset=None, optimize=True, **kwargs):
        """Apply a function to the axes listed in axis list.
        The function must operate on numpy arrays. It is applied to the data at
        each coordinate pixel-py-pixel according to axis list.
        Parameters
        ----------
        function : :std:term:`function`
            Any function that can be applied to the signal.
        inplace : bool
            if ``True`` (default), the data is replaced by the result. Otherwise
            a new Signal with the results is returned.
        ragged : None or bool
            Indicates if the results for each navigation pixel are of identical
            shape (and/or numpy arrays to begin with). If ``None``,
            the appropriate choice is made while processing. Note: ``None``
            is not allowed for Lazy signals!
        **kwargs : dict
            All extra keyword arguments are passed to the provided function
        """
        # Sepate ndkwargs
        ndkwargs = ()
        for key, value in kwargs.items():
            if isinstance(value, BaseSignal):
                ndkwargs += ((key, value),)

        res = self._map_iterate(function, iterating_kwargs=ndkwargs,
                                show_progressbar=show_progressbar,
                                parallel=parallel, inplace=inplace,
                                ragged=ragged,
                                is_navigation=is_navigation,
                                **kwargs)
        if inplace:
            if is_navigation:
                self.transpose(optimize=optimize)
            if scale is not None:
                for a, s in enumerate(scale):
                    self.axes_manager[a].scale = s
            if units is not None:
                for a,u in enumerate(units):
                    print("The axis index is:", a)
                    self.axes_manager[a].units = u
            if offset is not None:
                for a,o in enumerate(offset):
                    print("The axis index is:", a)
                    self.axes_manager[a].offset = o
            self.events.data_changed.trigger(obj=self)
        else:
            if is_navigation:
                res.transpose(optimize=optimize)
            if scale is not None:
                for a, s in enumerate(scale):
                    res.axes_manager[a].scale = s
            if units is not None:
                for a, u in enumerate(units):
                    res.axes_manager[a].units = u
            if offset is not None:
                for a, o in enumerate(offset):
                    res.axes_manager[a].offset = o
        return res

    def as_lazy(self, *args, **kwargs):
        """ Change the signal to lazy loading signal.  For large signals.
        """
        res = super().as_lazy(*args, **kwargs)
        res.__class__ = LazyAmorphousSignal
        res.__init__(**res._to_dictionary())
        return res

    def center_direct_beam(self, center):
        """Center the direct beam.  Shifts the center by shifting the offset so the center is in the middle.

        Parameters
        --------------
        center: tuple
            The x and y coordinates of the center. Either as integers or real float values.
        """
        if not all(isinstance(item, int) for item in center):
            center = (self.axes_manager.signal_axes[1].value2index(center[1]),
                      self.axes_manager.signal_axes[0].value2index(center[0]))
        self.axes_manager.signal_axes[0].offset = -center[0]
        self.axes_manager.signal_axes[1].offset = -center[1]

    def estimate_distortion(self, **kwargs):
        masked_sum = self.metadata.Sum.sig_sum.data
        print(self)
        print(np.shape(self.metadata.Mask.sig_mask))
        print(np.shape(self.metadata.Sum.sig_sum.data))
        center, lengths, angle = solve_ellipse(masked_sum, mask=self.metadata.Mask.sig_mask, **kwargs)
        return center, lengths, angle

    def get_thicknesses(self):
        """ Returns a Signal2D object with the thicknesses based on the High Angular annular detector and the
        applied calibration.
        """
        if self.metadata.HAADF.filter_slope and self.metadata.HAADF.filter_intercept:
            return self.metadata.HAADF.intensity*self.metadata.HAADF.filter_slope+self.metadata.HAADF.filter_intercept
        else:
            print("You need a slope and an intercept to get the thicknesses from the High Angle Annular Dark field "
                  "Image")
            return

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

    def thickness_filter(self):
        """Filter based on HAADF intensities.  Requires that there is a HAADF Signal in the metadata.

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
                     np.mean(thickness) + twosigma/2,
                     np.mean(thickness) + 3 * twosigma / 2]
        return th_filter, thickness

    def to_polar(self, center=None, lengths=None, angle=None, phase_width=None, radius=[0, -1],
                 estimate_distortion=False, inplace=False, normalize=True,  **kwargs):
        if not phase_width:
            phase_width = round((self.axes_manager.signal_shape[0]*3.14/2)/180)*180
        if phase_width is 0:
            phase_width = 90
        if estimate_distortion:
            center, lengths, angles = self.estimate_distortion()
        else:
            if isinstance(radius[0], float) or isinstance(radius[1], float):
                radius[0] = self.axes_manager.signal_axes[-1].value2index(radius[0])
                radius[1] = self.axes_manager.signal_axes[-1].value2index(radius[1])
            if radius[1] == -1:
                if center is None:
                    radius[1] = int(min(self.axes_manager.signal_shape) - 2)
                else:
                    radius[1] = int(min(np.subtract(self.axes_manager.signal_shape, center)) - 1)
        print(phase_width)
        polar_signal = self.axis_map(to_polar_image,
                                     center=center,
                                     angle=angle,
                                     lengths=lengths,
                                     phase_width=phase_width,
                                     radius=radius,
                                     inplace=inplace,
                                     normalize=normalize,
                                     **kwargs,
                                     scale=[(2 * np.pi/phase_width), self.axes_manager.signal_axes[1].scale],
                                     units=["Radians,$\Theta$", "$nm^-1$"])
        polar_mask = to_polar_image(self.metadata.Mask.sig_mask, angle=angle, lengths=lengths, radius=radius,
                                    phase_width=phase_width)
        polar_mask = polar_mask > 0
        if inplace:
            self.masig = polar_mask
            self.set_signal_type("PolarAmorphous2D")
        else:
            polar_signal.masig = polar_mask
            polar_signal.set_signal_type("PolarAmorphous2D")
            print(polar_signal)

        return polar_signal

    def _iterate_navigation(self):
        """Iterates over the navigation data.
        It is faster than using the navigation iterator.
        """
        self._make_sure_data_is_contiguous()
        axes = [axis.index_in_array for axis in self.axes_manager.navigation_axes]
        if axes:
            unfolded_axis = self.axes_manager.signal_axes[0].index_in_array
            new_shape = [1] * len(self.data.shape)
            for axis in axes:
                new_shape[axis] = self.data.shape[axis]
            new_shape[unfolded_axis] = -1
        else:  # navigation_dimension == 0
            new_shape = (1,-1)
            axes = [1]
            unfolded_axis = 0
        # Warning! if the data is not contigous it will make a copy!!
        print(new_shape)
        data = self.data.reshape(new_shape)
        print(data.shape)
        getitem = [0] * len(data.shape)
        for axis in axes:
            getitem[axis] = slice(None)
        for i in range(data.shape[unfolded_axis]):
            getitem[unfolded_axis] = i
            yield data[tuple(getitem)]

    def _map_iterate(self, function, iterating_kwargs=(),
                     show_progressbar=None, parallel=None,
                     ragged=None,
                     inplace=True,
                     is_navigation=False,
                     **kwargs):
        """Iterates the signal navigation space applying the function.
        Parameters
        ----------
        function : :std:term:`function`
            the function to apply
        iterating_kwargs : tuple (of tuples)
            A tuple with structure (('key1', value1), ('key2', value2), ..)
            where the key-value pairs will be passed as kwargs for the
            function to be mapped, and the values will be iterated together
            with the signal navigation.
        %s
        %s
        inplace : bool
            if ``True`` (default), the data is replaced by the result. Otherwise
            a new signal with the results is returned.
        ragged : None or bool
            Indicates if results for each navigation pixel are of identical
            shape (and/or numpy arrays to begin with). If ``None``,
            an appropriate choice is made while processing. Note: ``None`` is
            not allowed for Lazy signals!
        **kwargs : dict
            Additional keyword arguments are passed to `function`
        Notes
        -----
        This method is replaced for lazy signals.
        Examples
        --------
        Pass a larger array of different shape
        >>> s = hs.signals.Signal1D(np.arange(20.).reshape((20,1)))
        >>> def func(data, value=0):
        ...     return data + value
        >>> # pay attention that it's a tuple of tuples - need commas
        >>> s._map_iterate(func,
        ...                iterating_kwargs=(('value',
        ...                                    np.random.rand(5,400).flat),))
        >>> s.data.T
        array([[  0.82869603,   1.04961735,   2.21513949,   3.61329091,
                  4.2481755 ,   5.81184375,   6.47696867,   7.07682618,
                  8.16850697,   9.37771809,  10.42794054,  11.24362699,
                 12.11434077,  13.98654036,  14.72864184,  15.30855499,
                 16.96854373,  17.65077064,  18.64925703,  19.16901297]])
        Storing function result to other signal (e.g. calculated shifts)
        >>> s = hs.signals.Signal1D(np.arange(20.).reshape((5,4)))
        >>> def func(data): # the original function
        ...     return data.sum()
        >>> result = s._get_navigation_signal().T
        >>> def wrapped(*args, data=None):
        ...     return func(data)
        >>> result._map_iterate(wrapped,
        ...                     iterating_kwargs=(('data', s),))
        >>> result.data
        array([  6.,  22.,  38.,  54.,  70.])
        """
        if parallel is None:
            parallel = preferences.General.parallel
        if parallel is True:
            from os import cpu_count
            parallel = cpu_count() or 1
        # Because by default it's assumed to be I/O bound, and cpu_count*5 is
        # used. For us this is not the case.

        if show_progressbar is None:
            show_progressbar = preferences.General.show_progressbar
        if is_navigation:
            size = max(1, self.axes_manager.signal_size)
        else:
            size = max(1, self.axes_manager.navigation_size)

        from hyperspy.misc.utils import (create_map_objects)
        func, iterators = create_map_objects(function, size, iterating_kwargs,
                                             **kwargs)
        if is_navigation:
            iterators = (self._iterate_navigation(),) + iterators
            res_shape = self.axes_manager._signal_shape_in_array
        else:
            iterators = (self._iterate_signal(),) + iterators
            res_shape = self.axes_manager._navigation_shape_in_array
        if not len(res_shape):
            res_shape = (1,)
        # pre-allocate some space
        res_data = np.empty(res_shape, dtype='O')
        shapes = set()

        # parallel or sequential maps
        if parallel:
            from concurrent.futures import ThreadPoolExecutor
            executor = ThreadPoolExecutor(max_workers=parallel)
            thismap = executor.map
        else:
            from builtins import map as thismap
        pbar = progressbar(total=size, leave=True, disable=not show_progressbar)
        for ind, res in zip(range(res_data.size),
                            thismap(func, zip(*iterators))):
            # In what follows we assume that res is a numpy scalar or array
            # The following line guarantees that that's the case.
            res = np.asarray(res)
            res_data.flat[ind] = res
            if ragged is False:
                # to be able to break quickly and not waste time / resources
                shapes.add(res.shape)
                if len(shapes) != 1:
                    raise ValueError('The result shapes are not identical, but'
                                     'ragged=False')
            else:
                try:
                    shapes.add(res.shape)
                except AttributeError:
                    shapes.add(None)
            pbar.update(1)
        if parallel:
            executor.shutdown()

        # Combine data if required
        shapes = list(shapes)

        suitable_shapes = len(shapes) == 1 and shapes[0] is not None
        ragged = ragged or not suitable_shapes
        sig_shape = None
        if not ragged:
            sig_shape = () if shapes[0] == (1,) else shapes[0]
            if is_navigation:
                res_data = np.stack(res_data.ravel(), axis=2).reshape(sig_shape + self.axes_manager._signal_shape_in_array)
                # Im not sure why the axis needs to be two.. I mean I know it has to do with how the axes need to be
                # oriented but if you had 3 navigation axes it would probably have to be 3.. I'll figure out later
            else:
                res_data = np.stack(res_data.ravel()).reshape(self.axes_manager._navigation_shape_in_array + sig_shape)
        res = map_result_construction(self, inplace, res_data, ragged,
                                      sig_shape, is_navigation)
        return res

    _map_iterate.__doc__ %= (SHOW_PROGRESSBAR_ARG, PARALLEL_ARG)

    def get_virtual_image(self, roi):
        dark_field = roi(self, axes=self.axes_manager.signal_axes)
        dark_field_sum = dark_field.sum(
            axis=dark_field.axes_manager.signal_axes
        )
        dark_field_sum.metadata.General.title = "Virtual Dark Field"
        vdfim = dark_field_sum.as_signal2D((0, 1))
        return vdfim


class LazyAmorphousSignal(LazySignal, Amorphous2D):
    _lazy = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _map_iterate(self,
                     function,
                     iterating_kwargs=(),
                     show_progressbar=None,
                     parallel=None,
                     ragged=None,
                     inplace=True,
                     is_navigation=False,
                     **kwargs):
        if ragged not in (True, False):
            raise ValueError('"ragged" kwarg has to be bool for lazy signals')
        if is_navigation:
            size = max(1, self.axes_manager.signal_size)
        else:
            size = max(1, self.axes_manager.navigation_size)
        from hyperspy.misc.utils import (create_map_objects,
                                         map_result_construction)
        func, iterators = create_map_objects(function, size, iterating_kwargs,
                                             **kwargs)
        if is_navigation:
            iterators = (self._iterate_navigation(), ) + iterators
        else:
            iterators = (self._iterate_signal(), ) + iterators
        if is_navigation:
            res_shape = self.axes_manager._signal_shape_in_array
        else:
            res_shape = self.axes_manager._navigation_shape_in_array
        # no navigation
        if not len(res_shape) and ragged:
            res_shape = (1,)

        all_delayed = [dd(func)(data) for data in zip(*iterators)]

        if ragged:
            sig_shape = ()
            sig_dtype = np.dtype('O')
        else:
            one_compute = all_delayed[0].compute()
            sig_shape = one_compute.shape
            sig_dtype = one_compute.dtype
        pixels = [
            da.from_delayed(
                res, shape=sig_shape, dtype=sig_dtype) for res in all_delayed
        ]

        for step in reversed(res_shape):
            _len = len(pixels)
            starts = range(0, _len, step)
            ends = range(step, _len + step, step)
            if is_navigation:
                pixels = [da.stack(pixels[s:e], axis=2) for s, e in zip(starts, ends)]
            else:
                pixels = [da.stack(pixels[s:e], axis=0) for s, e in zip(starts, ends)]
        result = pixels[0]
        res = map_result_construction(
            self, inplace, result, ragged, sig_shape, lazy=True)
        return res

    def _iterate_navigation(self):
        if self.axes_manager.signal_size < 2:
            yield self()
            return
        nav_dim = self.axes_manager.navigation_dimension
        sig_dim = self.axes_manager.signal_dimension
        sig_indices = self.axes_manager.signal_indices_in_array[::-1]
        sig_lengths = np.atleast_1d(
            np.array(self.data.shape)[list(sig_indices)])
        getitem = [slice(None)] * (nav_dim + sig_dim)
        data = self._lazy_data()
        for indices in product(*[range(l) for l in sig_lengths]):
            for res, ind in zip(indices, sig_indices):
                getitem[ind] = res
            yield data[tuple(getitem)]