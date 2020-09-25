import gunpowder as gp


class RemoveChannelDim(gp.BatchFilter):
    """
        Squeezes the specified channel dim from the array.

        Args:

            array (:class: `ArrayKey`):

                The Array to remove a dim from.

            axis (:class: `int`):

                The axis to remove, must be of size 1.
    """

    def __init__(self, array, axis=0):
        self.array = array
        self.axis = axis

    def process(self, batch, request):

        if self.array not in batch:
            return
        data = batch[self.array].data
        shape = data.shape
        roi = batch[self.array].spec.roi

        assert self.axis < len(shape) - roi.dims(), \
            "Axis given not is in ROI and not channel dim, " \
            "Shape:" + str(shape) + " ROI: " + str(roi)
        assert shape[self.axis] == 1, \
            "Channel to delete must be size 1," \
            "but given shape " + str(shape)
        shape = self.__remove_dim(shape, self.axis)
        batch[self.array].data = data.reshape(shape)

    def __remove_dim(self, a, dim=0):
        return a[:dim] + a[dim + 1:]
