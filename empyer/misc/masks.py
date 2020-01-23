from hyperspy.misc.slicing import SpecialSlicers


class MaskSlicer(SpecialSlicers):
    """
    Expansion of the Special Slicer class. Used for applying a mask
    """
    def __setitem__(self, key, value):
        if isinstance(self.obj, MaskPasser):
            array_slices = self.obj.signal._get_array_slices(key, self.isNavigation)
            if self.isNavigation == self.obj.isNavigation:
                print("You can't used masig or manav twice")
            self.obj.signal.add_mask()
            array_slices = tuple([slice1 if not (slice1 == slice(None, None, None)) else slice2 for
                                  slice1, slice2 in zip(self.obj.slice, array_slices)])
            self.obj.signal.data.mask[array_slices] = value
        else:
            array_slices = self.obj._get_array_slices(key, self.isNavigation)
            self.obj.add_mask()
            self.obj.data.mask[array_slices] = value

    def __getitem__(self, key, out=None):
        if isinstance(self.obj, MaskPasser):
            if self.isNavigation == self.obj.isNavigation:
                print("You can't used masig or manav twice")
                return
            array_slices = self.obj.signal._get_array_slices(key, self.isNavigation)
            array_slices = tuple([slice1 if not (slice1 == slice(None, None, None)) else slice2 for
                                  slice1, slice2 in zip(self.obj.slice, array_slices)])
            return MaskPasser(self.obj.signal, array_slices, self.isNavigation)
        else:
            array_slices = self.obj._get_array_slices(key, self.isNavigation)
            return MaskPasser(self.obj, array_slices, self.isNavigation)


class MaskPasser():
    def __init__(self, s, sl, nav):
        self.signal = s
        self.slice = sl
        self.isNavigation = nav
        self.manav = MaskSlicer(self, isNavigation=True)
        self.masig = MaskSlicer(self, isNavigation=False)

    def mask_circle(self, center, radius, unmask=False):
        # TODO: Add more shapes
        """Applies a mask to every pixel using a shape and the appropriate definition

        Parameters
        ----------
        center: tuple
            The (x,y) center of the circle
        radius: float or int
            The radius of the circle
        unmask: bool
            Unmask any pixels in the defined shape
        """
        self.signal.add_mask()
        if not all(isinstance(item, int) for item in center):
            center = (self.signal.axes_manager.signal_axes[1].value2index(center[1]),
                     self.signal.axes_manager.signal_axes[0].value2index(center[0]))
        if not isinstance(radius, int):
            radius = self.signal.axes_manager.signal_axes[0].value2index(radius)
        x_ind, y_ind = np.meshgrid(range(-radius, radius + 1), range(-radius, radius + 1))
        r = np.sqrt(x_ind ** 2 + y_ind ** 2)
        inside = r < radius
        x_ind, y_ind = x_ind[inside]+int(center[0]), y_ind[inside]+int(center[1])
        self.signal.data.mask[self.slice][..., x_ind, y_ind] = not unmask
        return
