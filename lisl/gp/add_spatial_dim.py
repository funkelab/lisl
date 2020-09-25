import gunpowder as gp
import numpy as np


class AddSpatialDim(gp.BatchFilter):
    """
        Adds a spatial dim of size 1 to the begining of the ROI.
        Requests an upstream ROI with the first dim removed.

        Currently only works for Arrays.

        Args:

            array (:class: ArrayKey):

                The array key to modify.
    """

    def __init__(self, array):
        self.array = array

    def setup(self):

        upstream_spec = self.get_upstream_provider().spec[self.array]

        spec = upstream_spec.copy()
        if spec.roi is not None:
            spec.roi = gp.Roi(
                self.__insert_dim(spec.roi.get_begin(), 0),
                self.__insert_dim(spec.roi.get_shape(), 1))
        if spec.voxel_size is not None:
            spec.voxel_size = self.__insert_dim(spec.voxel_size, 1)
        self.spec[self.array] = spec
        self.updates(self.array, self.spec[self.array])

    def prepare(self, request):

        if self.array not in request:
            return

        request[self.array].roi = gp.Roi(
            self.__remove_dim(request[self.array].roi.get_begin()),
            self.__remove_dim(request[self.array].roi.get_shape()))
        if request[self.array].voxel_size is not None:
            request[self.array].voxel_size = self.__remove_dim(
                request[self.array].voxel_size)

    def process(self, batch, request):

        if self.array not in request:
            return

        array = batch[self.array]

        array.spec.roi = gp.Roi(
            self.__insert_dim(array.spec.roi.get_begin(), 0),
            self.__insert_dim(array.spec.roi.get_shape(), 1))
        array.spec.voxel_size = self.__insert_dim(array.spec.voxel_size, 1)
        array.data = array.data[:, :, np.newaxis, :, :]

    def __remove_dim(self, a, dim=0):
        return a[:dim] + a[dim + 1:]

    def __insert_dim(self, a, s, dim=0):
        return a[:dim] + (s,) + a[dim:]
