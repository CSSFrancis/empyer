import numpy as np

from hyperspy._signals.signal2d import Signal2D
from hyperspy.signal import BaseSignal
from hyperspy._signals.lazy import LazySignal
from empyer.misc.masks import Mask
from empyer.misc.ellipse_analysis import solve_ellipse
from empyer.misc.cartesain_to_polar import convert
from hyperspy.defaults_parser import preferences
from hyperspy.docstrings.signal import SHOW_PROGRESSBAR_ARG, PARALLEL_ARG
from hyperspy.external.progressbar import progressbar


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
        self.manav = Mask(self,is_navigation=True)
        self.masig = Mask(self,is_navigation=False)
        if not self.metadata.has_item('Mask'):
            self.metadata.add_node('Mask.sig_mask')
            self.metadata.add_node("Mask.nav_mask")
            self.metadata.Mask.sig_mask = np.zeros(shape=self.axes_manager.signal_shape, dtype=bool)
            self.metadata.Mask.nav_mask = np.zeros(shape=self.axes_manager.navigation_shape, dtype=bool)

        if not self.metadata.has_item('Sum'):
            self.metadata.add_node("Sum.sig_sum")
            self.metadata.add_node("Sum.nav_sum")
            self.metadata.Sum.sig_sum = self.sum(axis=self.axes_manager.signal_axes)
            self.metadata.Mask.nav_sum = self.sum(axis=self.axes_manager.navigation_axes)

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
            show_progressbar=None, parallel=None, inplace=True, ragged=None, is_navigation=False, scale=None, units=None,
            optimize=True, **kwargs):
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

        # Check if the signal axes have inhomogeneous scales and/or units and
        # display in warning if yes.
        if is_navigation:
            axes_index =[ a.index for a in self.axes_manager.navigation_axes]
        else:
            axes_index =[a.index for a in self.axes_manager.signal_axes]
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
                for s, a in scale, axes_index:
                    self.axes_manager[a].scale = s
            if units is not None:
                for u, a in units, axes_index:
                    self.axes_manager[a].units = u
            self.events.data_changed.trigger(obj=self)
        else:
            if is_navigation:
                res.transpose(optimize=optimize)
            if scale is not None:
                for s, a in scale, axes_index:
                    res.axes_manager[a].scale = s
            if units is not None:
                for u, a in units, axes_index:
                    res.axes_manager[a].units = u
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
        masked_sum = self.metadata.Sum.sig_sum
        masked_sum[self.masig] = 0
        center, lengths, angle = solve_ellipse(masked_sum, **kwargs)
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
        """Filter based on HAADF intensities

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

    def to_polar(self, center=None, lengths=None, angle=None, phase_width=None, radius=[0,-1],**kwargs):
        if not phase_width:
            phase_width = round((self.axes_manager.signal_shape[0]*3.14/2)/180)*180
        if isinstance(radius[0], float) or isinstance(radius[1], float):
            radius[0] = self.axes_manager.signal_axes[-1].value2index(radius[0])
            radius[1] = self.axes_manager.signal_axes[-1].value2index(radius[1])
        if radius[1] == -1:
            radius[1] = int(min(np.subtract(self.axes_manager.signal_shape, center)) - 1)

        polar_signal = self.axis_map(convert,
                                     center=self.metadata.Signal.Ellipticity.center,
                                     angle=self.metadata.Signal.Ellipticity.angle,
                                     lengths=self.metadata.Signal.Ellipticity.lengths,
                                     phase_width=phase_width,
                                     radius=radius,
                                     **kwargs,
                                     scale=[(2 * np.pi/phase_width),self.axes_manager.signal_axes[1].scale],
                                     units=["Radians,$\Theta$", "$nm^-1$"])

        return polar_signal

    def _iterate_navigation(self):
        """Iterates over the navigation data.
        It is faster than using the navigation iterator.
        """
        self._make_sure_data_is_contiguous()
        axes = [axis.index_in_array for
                axis in self.axes_manager.navigation_axes]
        if axes:
            unfolded_axis = self.axes_manager.signal_axes[0].index_in_array
            new_shape = [1] * len(self.data.shape)
            for axis in axes:
                new_shape[axis] = self.data.shape[axis]
            new_shape[unfolded_axis] = -1
        else:  # signal_dimension == 0
            new_shape = (-1, 1)
            axes = [1]
            unfolded_axis = 0
        # Warning! if the data is not contigous it will make a copy!!
        data = self.data.reshape(new_shape)
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

        from hyperspy.misc.utils import (create_map_objects,
                                         map_result_construction)
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
                res_data = np.stack(res_data.ravel()).reshape(sig_shape + self.axes_manager._signal_shape_in_array)
            else:
                res_data = np.stack(res_data.ravel()).reshape(self.axes_manager._navigation_shape_in_array + sig_shape)
        res = map_result_construction(self, inplace, res_data, ragged,
                                      sig_shape)
        return res

    _map_iterate.__doc__ %= (SHOW_PROGRESSBAR_ARG, PARALLEL_ARG)

    def to_correlation(self):
        pass

    def get_virtual_image(self, roi):
        return


class LazyAmorphousSignal(LazySignal, Amorphous2D):
    _lazy = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def axis_map(self, function, is_navigation=False, **kwargs):
        """Applies a 2-D filter on either the navigation or the signal axis
        Parameters
        --------------
        axes: list
            The indexes of the axes to be operated on for the 2D transformation
        function:function
            Any function that can be applied to a 2D signal
        show_progressbar: (None or bool)
            If True, display a progress bar. If None, the default from the preferences settings is used.
        parallel:(None or bool)
            If True, perform computation in parallel using multiple cores. If None, the default from the preferences
            settings is used.
        inplace: bool
            if True (default), the data is replaced by the result. Otherwise a new Signal with the results is returned.
        ragged:(None or bool)
            Indicates if the results for each navigation pixel are of identical shape (and/or numpy arrays to begin with). If None, the appropriate choice is made while processing. Note: None is not allowed for Lazy signals!
        is_navigation:(bool)
            If the function should be operating on the navigation axes rather than the signal axis.
        **kwargs (dict)
            All extra keyword arguments are passed to the provided function
        """
        if is_navigation:
            return self.Transpose(optimze=True).map(function, **kwargs)
        else:
            return self.map(function, **kwargs)
        return
