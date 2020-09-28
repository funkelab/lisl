from .augmentations import add_augmentation_pipeline
from lisl.gp import (
    AddChannelDim,
    AddSpatialDim,
    RandomBranchSelector,
    RandomPointGenerator,
    RandomPointSource,
    RandomProvider,
    RejectArray,
    RemoveChannelDim)
import daisy
import gunpowder as gp
import logging
import numpy as np
import os

logger = logging.getLogger(__name__)


class PrepareBatch(gp.BatchFilter):
    """Prepares a batch of with pairs of points for use in loss functions.

    Finds the intersecting nodes, and then converts them to unit locations by
    subtracting the offset and dividing by raw's voxel size. Also adds the
    batch shape to the points.

    If the locations should be 2D but aren't the is_2d flag should be true and
    will only return the last 2 dims of the locations.
    """
    def __init__(
            self,
            raw_0, raw_1,
            points_0, points_1,
            locations_0, locations_1,
            is_2d):
        self.raw_0 = raw_0
        self.raw_1 = raw_1
        self.points_0 = points_0
        self.points_1 = points_1
        self.locations_0 = locations_0
        self.locations_1 = locations_1
        self.is_2d = is_2d

    def setup(self):
        self.provides(
            self.locations_0,
            gp.ArraySpec(nonspatial=True))
        self.provides(
            self.locations_1,
            gp.ArraySpec(nonspatial=True))

    def process(self, batch, request):

        ids_0 = set([n.id for n in batch[self.points_0].nodes])
        ids_1 = set([n.id for n in batch[self.points_1].nodes])
        common_ids = ids_0.intersection(ids_1)

        locations_0 = []
        locations_1 = []
        # get list of only xy locations
        # locations are in voxels, relative to output roi
        points_roi = request[self.points_0].roi
        voxel_size = batch[self.raw_0].spec.voxel_size
        for i in common_ids:
            location_0 = np.array(batch[self.points_0].node(i).location)
            location_1 = np.array(batch[self.points_1].node(i).location)
            if not points_roi.contains(location_0):
                print(f"skipping point {i} at {location_0}")
                continue
            if not points_roi.contains(location_1):
                print(f"skipping point {i} at {location_1}")
                continue
            location_0 -= points_roi.get_begin()
            location_1 -= points_roi.get_begin()
            location_0 /= voxel_size
            location_1 /= voxel_size
            locations_0.append(location_0)
            locations_1.append(location_1)

        locations_0 = np.array(locations_0, dtype=np.float32)
        locations_1 = np.array(locations_1, dtype=np.float32)
        if self.is_2d:
            locations_0 = locations_0[:, 1:]
            locations_1 = locations_1[:, 1:]
        locations_0 = locations_0[np.newaxis]
        locations_1 = locations_1[np.newaxis]

        # create point location arrays (with batch dimension)
        batch[self.locations_0] = gp.Array(
            locations_0, self.spec[self.locations_0])
        batch[self.locations_1] = gp.Array(
            locations_1, self.spec[self.locations_1])

        # add batch dimension to raw
        batch[self.raw_0].data = batch[self.raw_0].data[np.newaxis, :]
        batch[self.raw_1].data = batch[self.raw_1].data[np.newaxis, :]

        # make sure raw is float32
        batch[self.raw_0].data = batch[self.raw_0].data.astype(np.float32)
        batch[self.raw_1].data = batch[self.raw_1].data.astype(np.float32)


def random_point_pairs_pipeline(
        model, loss, optimizer,
        dataset,
        augmentation_parameters,
        point_density,
        out_dir,
        normalize_factor=None,
        checkpoint_interval=10000,
        snapshot_interval=500):

    raw_0 = gp.ArrayKey('RAW_0')
    points_0 = gp.GraphKey('POINTS_0')
    locations_0 = gp.ArrayKey('LOCATIONS_0')
    emb_0 = gp.ArrayKey('EMBEDDING_0')
    raw_1 = gp.ArrayKey('RAW_1')
    points_1 = gp.GraphKey('POINTS_1')
    locations_1 = gp.ArrayKey('LOCATIONS_1')
    emb_1 = gp.ArrayKey('EMBEDDING_1')

    data = daisy.open_ds(dataset.filename, dataset.ds_names[0])
    source_roi = gp.Roi(data.roi.get_offset(), data.roi.get_shape())
    voxel_size = gp.Coordinate(data.voxel_size)

    # Get in and out shape
    in_shape = gp.Coordinate(model.in_shape)
    out_shape = gp.Coordinate(model.out_shape[2:])
    is_2d = in_shape.dims() == 2

    emb_voxel_size = voxel_size

    # Add fake 3rd dim
    if is_2d:
        in_shape = gp.Coordinate((1, *in_shape))
        out_shape = gp.Coordinate((1, *out_shape))
        voxel_size = gp.Coordinate((1, *voxel_size))
        source_roi = gp.Roi((0, *source_roi.get_offset()),
                            (data.shape[0], *source_roi.get_shape()))

    in_shape = in_shape * voxel_size
    out_shape = out_shape * voxel_size

    logger.info(f"source roi: {source_roi}")
    logger.info(f"in_shape: {in_shape}")
    logger.info(f"out_shape: {out_shape}")
    logger.info(f"voxel_size: {voxel_size}")

    request = gp.BatchRequest()
    request.add(raw_0, in_shape)
    request.add(raw_1, in_shape)
    request.add(points_0, out_shape)
    request.add(points_1, out_shape)
    request[locations_0] = gp.ArraySpec(nonspatial=True)
    request[locations_1] = gp.ArraySpec(nonspatial=True)

    snapshot_request = gp.BatchRequest()
    snapshot_request[emb_0] = gp.ArraySpec(roi=request[points_0].roi)
    snapshot_request[emb_1] = gp.ArraySpec(roi=request[points_1].roi)

    random_point_generator = RandomPointGenerator(
        density=point_density, repetitions=2)

    # Use volume to calculate probabilities, RandomBranchSelector will
    # normalize volumes to probablilties
    probabilities = np.array([
        np.product(daisy.open_ds(dataset.filename, ds_name).shape)
        for ds_name in dataset.ds_names
    ])
    random_source_generator = RandomBranchSelector(
        num_sources=len(dataset.ds_names),
        probabilities=probabilities,
        repetitions=2)

    array_sources = tuple(
        tuple(
            gp.ZarrSource(
                dataset.filename,
                {
                    raw: ds_name
                },
                # fake 3D data
                array_specs={
                    raw: gp.ArraySpec(
                        roi=source_roi,
                        voxel_size=voxel_size,
                        interpolatable=True)
                })

            for ds_name in dataset.ds_names
        )
        for raw in [raw_0, raw_1]
    )

    # Choose a random dataset to pull from
    array_sources = \
        tuple(arrays +
              RandomProvider(random_source_generator) +
              gp.Normalize(raw, normalize_factor) +
              gp.Pad(raw, None)
              for raw, arrays
              in zip([raw_0, raw_1], array_sources))

    point_sources = tuple((
        RandomPointSource(
            points_0,
            random_point_generator=random_point_generator),
        RandomPointSource(
            points_1,
            random_point_generator=random_point_generator)
    ))

    # Merge the point and array sources together.
    # There is one array and point source per branch.
    sources = tuple(
        (array_source, point_source) + gp.MergeProvider()
        for array_source, point_source
        in zip(array_sources, point_sources))

    sources = tuple(
        add_augmentation_pipeline(source, raw, **augmentation_parameters)
        for raw, source in zip([raw_0, raw_1], sources)
    )

    pipeline = (
        sources +
        gp.MergeProvider() +
        gp.Crop(raw_0, source_roi) +
        gp.Crop(raw_1, source_roi) +
        gp.RandomLocation() +
        PrepareBatch(
            raw_0, raw_1,
            points_0, points_1,
            locations_0, locations_1,
            is_2d) +
        RejectArray(ensure_nonempty=locations_0) +
        RejectArray(ensure_nonempty=locations_1))

    if not is_2d:
        pipeline = (
            pipeline +
            AddChannelDim(raw_0) +
            AddChannelDim(raw_1)
        )

    pipeline = (
        pipeline +
        gp.PreCache() +
        gp.torch.Train(
            model, loss, optimizer,
            inputs={
                'raw_0': raw_0,
                'raw_1': raw_1
            },
            loss_inputs={
                'emb_0': emb_0,
                'emb_1': emb_1,
                'locations_0': locations_0,
                'locations_1': locations_1
            },
            outputs={
                2: emb_0,
                3: emb_1
            },
            array_specs={
                emb_0: gp.ArraySpec(voxel_size=emb_voxel_size),
                emb_1: gp.ArraySpec(voxel_size=emb_voxel_size)
            },
            checkpoint_basename=os.path.join(
                out_dir,
                'checkpoint'),
            save_every=checkpoint_interval
        )
    )

    if is_2d:
        pipeline = (
            pipeline +
            # everything is 3D, except emb_0 and emb_1
            AddSpatialDim(emb_0) +
            AddSpatialDim(emb_1)
        )

    pipeline = (
        pipeline +
        # now everything is 3D
        RemoveChannelDim(raw_0) +
        RemoveChannelDim(raw_1) +
        RemoveChannelDim(emb_0) +
        RemoveChannelDim(emb_1) +
        gp.Snapshot(
            output_dir=os.path.join(
                out_dir,
                'snapshots'),
            output_filename='iteration_{iteration}.hdf',
            dataset_names={
                raw_0: 'raw_0',
                raw_1: 'raw_1',
                locations_0: 'locations_0',
                locations_1: 'locations_1',
                points_0: 'points_0',
                points_1: 'points_1',
                emb_0: 'emb_0',
                emb_1: 'emb_1'
            },
            additional_request=snapshot_request,
            every=snapshot_interval) +
        gp.PrintProfilingStats(every=500)
    )

    return pipeline, request
