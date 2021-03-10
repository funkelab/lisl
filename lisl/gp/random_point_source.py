import gunpowder as gp
import numpy as np


class RandomPointGenerator:

    def __init__(self, density=None, repetitions=1, num_points=None):
        '''Create random points in a provided ROI with the given density.

        Args:

            density (float):

                The expected number of points per world unit cube. If, for
                example, the ROI passed to `get_random_points(roi)` has a 2D
                size of (10, 10) and the density is 1.0, 100 uniformly
                distributed points will be returned.

            repetitions (int):

                Return the same list of points that many times. Note that in
                general only the first call will contain uniformly distributed
                points for the given ROI. Subsequent calls with a potentially
                different ROI will only contain the points that lie within that
                ROI.
        '''
        self.density = density
        self.repetitions = repetitions
        self.iteration = 0
        self.num_points = num_points

    def get_random_points(self, roi):
        '''Get a dictionary mapping point IDs to nD locations, uniformly
        distributed in the given ROI. If `repetitions` is larger than 1,
        previously sampled points will be reused that many times.
        '''

        ndims = roi.dims()
        volume = np.prod(roi.get_shape())

        if self.iteration % self.repetitions == 0:

            # create random points in the unit cube
            if self.num_points is None:
                self.points = np.random.random(
                    (int(self.density * volume), ndims))
            else:
                self.points = np.random.random((self.num_points, ndims))
            # scale and shift into requested ROI
            self.points *= np.array(roi.get_end() - roi.get_begin())
            self.points += roi.get_begin()

            ret = {i: point for i, point in enumerate(self.points)}

        else:

            ret = {}
            for i, point in enumerate(self.points):
                if roi.contains(point):
                    ret[i] = point

        self.iteration += 1
        return ret


class RandomPointSource(gp.BatchProvider):

    def __init__(
            self,
            graph_key,
            dims,
            density=None,
            random_point_generator=None):
        '''A source creating uniformly distributed points.

        Args:

            graph_key (:class:`GraphKey`):

                The graph key to provide.

            dims (int):
                dimenstion of output points

            density (float, optional):

                The expected number of points per world unit cube. If, for
                example, the ROI passed to `get_random_points(roi)` has a 2D
                size of (10, 10) and the density is 1.0, 100 uniformly
                distributed points will be returned.

                Only used if `random_point_generator` is `None`.

            random_point_generator (:class:`RandomPointGenerator`, optional):

                The random point generator to use to create points.

                One of `density` or `random_point_generator` has to be given.
        '''

        assert (density is not None) != (random_point_generator is not None), \
            "Exactly one of 'density' or 'random_point_generator' has to be " \
            "given"

        self.dims = dims

        self.graph_key = graph_key
        if density is not None:
            self.random_point_generator = RandomPointGenerator(density=density)
        else:
            self.random_point_generator = random_point_generator

    def setup(self):

        # provide points in an infinite ROI
        self.graph_spec = gp.GraphSpec(
            roi=gp.Roi(
                offset=(0, ) * self.dims,
                shape=(None, ) * self.dims))

        self.provides(self.graph_key, self.graph_spec)

    def provide(self, request):

        roi = request[self.graph_key].roi

        random_points = self.random_point_generator.get_random_points(roi)

        batch = gp.Batch()
        batch[self.graph_key] = gp.Graph(
            [gp.Node(id=i, location=l) for i, l in random_points.items()],
            [],
            gp.GraphSpec(roi=roi, directed=False))

        return batch
