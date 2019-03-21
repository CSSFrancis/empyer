from hyperspy.drawing._markers.rectangle import Rectangle


class Rectangle_Mask(Rectangle):
    def __init__(self):
        Rectangle.__init__(self)
        self.marker_properties['is_mask'] = True

    def to_mask(self):

