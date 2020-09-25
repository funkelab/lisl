from collections.abc import Iterable
import gunpowder as gp
import numpy as np
import skimage.filters as filters


class Blur(gp.BatchFilter):
    '''Blurs an array. Uses the scikit-image function skimage.filters.gaussian.
    See scikit-image documentation for more information.

    Args:

        array (:class:`ArrayKey`):

            The array to blur.

        sigma (``scalar or list``):

            The standard deviation to use for the Gaussian filter. If scalar it
            will be projected to match the number of ROI dims. If a list or
            numpy array, it must match the number of ROI dims.

    '''

    def __init__(self, array, sigma=1.0):
        self.array = array
        self.sigma = np.array(sigma)

    def setup(self):

        spec = self.spec[self.array]
        dims = self.spec[self.array].roi.dims()

        if isinstance(self.sigma, Iterable):
            assert len(self.sigma) == dims, \
                   ("Dimensions given for sigma ("
                   + str(len(self.sigma)) + ") is not equal to the ROI dims ("
                   + str(dims) + ")")
        else:
            self.sigma = np.array([self.sigma] * dims)

        self.filter_radius = gp.Coordinate(np.ceil(3 * self.sigma))

        self.enable_autoskip()
        self.updates(self.array, spec)

    def prepare(self, request):

        deps = gp.BatchRequest()
        spec = request[self.array].copy()

        grown_roi = spec.roi.grow(
            self.filter_radius,
            self.filter_radius)
        grown_roi = grown_roi.snap_to_grid(spec.voxel_size)
        spec.roi = grown_roi
        deps[self.array] = spec

        return deps

    def process(self, batch, request):

        array = batch.arrays[self.array]

        array.data = filters.gaussian(
            array.data,
            sigma=self.sigma,
            mode='constant',
            preserve_range=True,
            multichannel=False)

        batch = gp.Batch()
        batch[self.array] = array.crop(request[self.array].roi)

        return batch
