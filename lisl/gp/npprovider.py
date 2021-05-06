from gunpowder.array import Array
from gunpowder.array_spec import ArraySpec
from gunpowder.nodes import BatchProvider

class NpProvider(BatchProvider):
	"""A source of data loaded form a numpy array"""
	def __init__(self, array, shape):
		super().__init__()
		self.array = array
		self.shape = shape

    def setup(self):

        # provide points in an infinite ROI
        self.graph_spec = gp.GraphSpec(
            roi=gp.Roi(
                offset=(0, 0, 0),
                shape=self.shape)

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



    def provide(self, request):

        timing = Timing(self)
        timing.start()

        batch = Batch()

        with self._open_file(self.filename) as data_file:
            for (array_key, request_spec) in request.array_specs.items():
                logger.debug("Reading %s in %s...", array_key, request_spec.roi)

                voxel_size = self.spec[array_key].voxel_size

                # scale request roi to voxel units
                dataset_roi = request_spec.roi / voxel_size

                # shift request roi into dataset
                dataset_roi = dataset_roi - self.spec[array_key].roi.get_offset() / voxel_size

                # create array spec
                array_spec = self.spec[array_key].copy()
                array_spec.roi = request_spec.roi

                # add array to batch
                batch.arrays[array_key] = Array(
                    self.__read(data_file, self.datasets[array_key], dataset_roi),
                    array_spec)

        logger.debug("done")

        timing.stop()
        batch.profiling_stats.add(timing)

        return batch


class RandomPointSource(gp.BatchProvider):

    def __init__(
            self,
            graph_key,
            density=None,
            random_point_generator=None):
        '''A source creating uniformly distributed points.

        Args:

            graph_key (:class:`GraphKey`):

                The graph key to provide.

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

        self.graph_key = graph_key
        if density is not None:
            self.random_point_generator = RandomPointGenerator(density=density)
        else:
            self.random_point_generator = random_point_generator

    def setup(self):

        # provide points in an infinite ROI
        self.graph_spec = gp.GraphSpec(
            roi=gp.Roi(
                offset=(0, 0, 0),
                shape=(None, None, None)))

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
