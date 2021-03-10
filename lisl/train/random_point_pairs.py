from funlib.learn.torch.models import UNet
from gunpowder.torch import Train
import gunpowder as gp
import math
import numpy as np
import torch
import logging

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

def random_point_pairs_pipeline(
        model, loss, optimizer,
        dataset,
        augmentation_parameters,
        point_density,
        out_dir,
        normalize_factor=None,
        checkpoint_interval=5000,
        snapshot_interval=5000):

    raw_0 = gp.ArrayKey('RAW_0')
    points_0 = gp.GraphKey('POINTS_0')
    locations_0 = gp.ArrayKey('LOCATIONS_0')
    emb_0 = gp.ArrayKey('EMBEDDING_0')
    raw_1 = gp.ArrayKey('RAW_1')
    points_1 = gp.GraphKey('POINTS_1')
    locations_1 = gp.ArrayKey('LOCATIONS_1')
    emb_1 = gp.ArrayKey('EMBEDDING_1')


    # TODO parse this key from somewhere
    key = 'train/raw/0'

    data = daisy.open_ds(dataset.filename, key)
    source_roi = gp.Roi(data.roi.get_offset(), data.roi.get_shape())
    voxel_size = gp.Coordinate(data.voxel_size)
    emb_voxel_size = voxel_size

    # Get in and out shape
    in_shape = gp.Coordinate(model.in_shape)
    out_shape = gp.Coordinate(model.out_shape)

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


    # Let's hardcode this for now
    # TODO read actual number from zarr file keys
    n_samples = 447
    batch_size = 1
    dim = 2
    padding = (100, 100)

    sources = []
    for i in range(n_samples):

        ds_key = f'train/raw/{i}'
        image_sources = tuple(
            gp.ZarrSource(
                dataset.filename,
                {
                    raw: ds_key
                },
                {
                    raw: gp.ArraySpec(interpolatable=True, voxel_size=(1, 1))
                }) + gp.Pad(raw, None)
            for raw in [raw_0, raw_1])


        random_point_generator = RandomPointGenerator(
            density=point_density, repetitions=2)

        point_sources = tuple((
            RandomPointSource(
                points_0,
                dim,
                random_point_generator=random_point_generator),
            RandomPointSource(
                points_1,
                dim,
                random_point_generator=random_point_generator)
        ))

        # TODO: get augmentation parameters from some config file!
        points_and_image_sources = tuple(
            (img_source, point_source) + gp.MergeProvider() + \
            gp.SimpleAugment() + \
            gp.ElasticAugment(
                spatial_dims=2,
                control_point_spacing=(10, 10),
                jitter_sigma=(0.0, 0.0),
                rotation_interval=(0, math.pi/2)) + \
            gp.IntensityAugment(r,
                                scale_min=0.8,
                                scale_max=1.2,
                                shift_min=-0.2,
                                shift_max=0.2,
                                clip=False) + \
            gp.NoiseAugment(r, var=0.01, clip=False)
            for r, img_source, point_source
            in zip([raw_0, raw_1], image_sources, point_sources))



        sample_source = points_and_image_sources + gp.MergeProvider()

        data = daisy.open_ds(dataset.filename, ds_key)
        source_roi = gp.Roi(data.roi.get_offset(), data.roi.get_shape())
        sample_source += gp.Crop(raw_0, source_roi)
        sample_source += gp.Crop(raw_1, source_roi)
        sample_source += gp.Pad(raw_0, padding)
        sample_source += gp.Pad(raw_1, padding)
        sample_source += gp.RandomLocation()
        sources.append(sample_source)

    sources = tuple(sources)

    pipeline = sources + gp.RandomProvider()
    pipeline += gp.Unsqueeze([raw_0, raw_1])

    pipeline += PrepareBatch(raw_0, raw_1,
                             points_0, points_1,
                             locations_0, locations_1)
    
    # How does prepare batch relate to Stack?????
    pipeline += RejectArray(ensure_nonempty=locations_1)
    pipeline += RejectArray(ensure_nonempty=locations_0)

    # batch content
    # raw_0:          (1, h, w)
    # raw_1:          (1, h, w)
    # locations_0:    (n, 2)
    # locations_1:    (n, 2)

    pipeline += gp.Stack(batch_size)

    # batch content
    # raw_0:          (b, 1, h, w)
    # raw_1:          (b, 1, h, w)
    # locations_0:    (b, n, 2)
    # locations_1:    (b, n, 2)

    pipeline += gp.PreCache(num_workers=10)

    pipeline += gp.torch.Train(model, loss, optimizer,
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
                               checkpoint_basename=os.path.join(out_dir, 'model'),
                               save_every=checkpoint_interval)
    
    pipeline += gp.Snapshot({
        raw_0 : 'raw_0',  
        raw_1 : 'raw_1',  
        emb_0 : 'emb_0',  
        emb_1 : 'emb_1',  
        # locations_0 : 'locations_0',  
        # locations_1 : 'locations_1',  
        },
        every=snapshot_interval,
        additional_request=snapshot_request)

    return pipeline, request




class PrintDebug(gp.BatchFilter):

    def prepare(self, request):
        print("prepare", request)

    def process(self, batch, request):
        print("process", request)
        print("process", batch)

# logger = logging.getLogger(__name__)


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
            locations_0, locations_1):
        self.raw_0 = raw_0
        self.raw_1 = raw_1
        self.points_0 = points_0
        self.points_1 = points_1
        self.locations_0 = locations_0
        self.locations_1 = locations_1

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

        # create point location arrays
        batch[self.locations_0] = gp.Array(
            locations_0, self.spec[self.locations_0])
        batch[self.locations_1] = gp.Array(
            locations_1, self.spec[self.locations_1])

        # make sure raw is float32
        batch[self.raw_0].data = batch[self.raw_0].data.astype(np.float32)
        batch[self.raw_1].data = batch[self.raw_1].data.astype(np.float32)


