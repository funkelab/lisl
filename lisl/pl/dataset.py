from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from lisl.pl.utils import UpSample, offset_from_direction
from lisl.pl.utils import AbsolutIntensityAugment
import gunpowder as gp
import daisy
import numpy as np
import random
from skimage.io import imsave
import math
import pytorch_lightning as pl
import argparse
import time
import logging
import os

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class IntensityFilter(gp.nodes.BatchFilter):

    def __init__(self, key, channel, min_intensity=0.01, quantile=0.99, rejection_sampling=True):
        self.key = key
        self.channel = channel
        self.min_intensity = min_intensity
        self.quantile = quantile
        self.rejection_sampling = rejection_sampling

    def setup(self):
        self.upstream_provider = self.get_upstream_provider()

    def provide(self, request):

        is_good_batch = False
        while not is_good_batch:

            batch = self.upstream_provider.request_batch(request)

            q = np.quantile(batch[self.key].data[self.channel], self.quantile)
            if q > self.min_intensity:
                is_good_batch = True

        return batch

class IntensityDiffFilter(gp.nodes.BatchFilter):

    def __init__(self, key, channel, min_distance=0.5, quantiles=(0.05, 0.95), passthrough=0.05):
        self.key = key
        self.channel = channel
        self.min_distance = min_distance
        self.quantiles = quantiles
        self.passthrough = passthrough

    def setup(self):
        self.upstream_provider = self.get_upstream_provider()

    def provide(self, request):

        is_good_batch = False

        coutner = 0

        while not is_good_batch:
            batch = self.upstream_provider.request_batch(request)
            q = np.quantile(batch[self.key].data[self.channel], self.quantiles)
            coutner += 1
            if abs(q[1] - q[0]) > self.min_distance:
                is_good_batch = True
            else:
                s = random.random()
                is_good_batch = self.passthrough > s

        return batch


class SparseChannelDataset(Dataset):
    """
    Volume loader dataset using a gunpowder pipeline
    TODO: Generalize for an arbitrary gp pipeline
    """

    def __init__(self, filename,
                 key,
                 density=None,
                 channels=0,
                 shape=(16, 256, 256),
                 time_window=None,
                 add_sparse_mosaic_channel=True,
                 random_rot=False):

        self.filename = filename
        self.key = key
        self.shape = shape
        self.density = density
        self.raw = gp.ArrayKey('RAW_0')
        self.add_sparse_mosaic_channel = add_sparse_mosaic_channel
        self.random_rot = random_rot
        self.channels = channels

        data = daisy.open_ds(filename, key)

        if time_window is None:
            source_roi = gp.Roi(data.roi.get_offset(), data.roi.get_shape())
        else:
            offs = list(data.roi.get_offset())
            offs[1] += time_window[0]
            sh = list(data.roi.get_shape())
            offs[1] = time_window[1] - time_window[0]
            source_roi = gp.Roi(tuple(offs), tuple(sh))

        voxel_size = gp.Coordinate(data.voxel_size)

        self.pipeline = gp.ZarrSource(
            filename,
            {
                self.raw: key
            },
            array_specs={
                self.raw: gp.ArraySpec(
                    roi=source_roi,
                    voxel_size=voxel_size,
                    interpolatable=True)
            }) + gp.RandomLocation() + IntensityDiffFilter(self.raw, 0, min_distance=0.1, channels=Slice(None))

        # add  augmentations
        self.pipeline = self.pipeline + gp.ElasticAugment([40, 40],
                                                          [2, 2],
                                                          [0, math.pi / 2.0],
                                                          prob_slip=-1,
                                                          spatial_dims=2)



        self.pipeline.setup()
        np.random.seed(os.getpid() + int(time.time()))


    def __getitem__(self, index):

        request = gp.BatchRequest()
        if self.channels == 0:
            request.add(self.raw, (1, ) + tuple(self.shape))
        else:
            request.add(self.raw, (2, 1, ) + tuple(self.shape))
            
        # Todo: replace with self.pipeline.setup() ?
        # Todo: self.pipeline.internal_teardown()
        # with gp.build(self.pipeline):
        out = self.pipeline.request_batch(request)[self.raw].data

        if self.channels == 0:
            out = out[None]

        if self.density is None:
            num_samples = random.randint(10, 200)

            y = [random.randint(0, out.shape[-2] - 1) for _ in range(num_samples)]
            x = [random.randint(0, out.shape[-1] - 1) for _ in range(num_samples)]
            mask = np.zeros(out[0, 0].shape, dtype=np.float32)
            mask[..., y, x] = 1
        else:
            mask = np.random.binomial(1, self.density, size=out[0, 0].shape).astype(np.float32)

        ubiquitous = out[0:1, 0].astype(np.float32)
        mosiac = out[1:2, 0].astype(np.float32)

        if self.add_sparse_mosaic_channel:
            source = np.concatenate((ubiquitous, mask[None], mask[None] * mosiac + 0.5 * (1 - mask[None])), axis=0)

        return ubiquitous, mosiac

    def __len__(self):
        # return a random length
        # determined by fair dice roll ;)
        # https://xkcd.com/221/
        return 8192

class RandomShiftDataset(Dataset):

    def __init__(self, filename,
                 key,
                 shape=(256, 256),
                 time_window=None,
                 max_direction=8,
                 distance=16,
                 upsample=None):

        self.filename = filename
        self.key = key
        self.shape = shape
        self.max_direction = max_direction
        self.distance = distance
        self.raw_0 = gp.ArrayKey('RAW_0')
        self.raw_1 = gp.ArrayKey('RAW_1')
        self.raw_0_us = gp.ArrayKey('RAW_0_US')
        self.raw_1_us = gp.ArrayKey('RAW_1_US')

        data = daisy.open_ds(filename, key)

        if time_window is None:
            source_roi = gp.Roi(data.roi.get_offset(), data.roi.get_shape())
        else:
            offs = list(data.roi.get_offset())
            offs[1] += time_window[0]
            sh = list(data.roi.get_shape())
            offs[1] = time_window[1] - time_window[0]
            source_roi = gp.Roi(tuple(offs), tuple(sh))

        voxel_size = gp.Coordinate(data.voxel_size)

        self.pipeline = gp.ZarrSource(
            filename,
            {
                self.raw_0: key,
                self.raw_1: key
            },
            array_specs={
                self.raw_0: gp.ArraySpec(
                    roi=source_roi,
                    voxel_size=voxel_size,
                    interpolatable=True),
                self.raw_1: gp.ArraySpec(
                    roi=source_roi,
                    voxel_size=voxel_size,
                    interpolatable=True)
            }) + gp.RandomLocation() + IntensityDiffFilter(self.raw_0, min_distance=0.1, channel=0)

        # add  augmentations
        self.pipeline = self.pipeline + gp.ElasticAugment([40, 40],
                                                          [2, 2],
                                                          [0, math.pi / 2.0],
                                                          prob_slip=-1,
                                                          spatial_dims=2)

        self.pipeline = self.pipeline + AbsolutIntensityAugment(self.raw_0,
                                                                   scale_min=0.7,
                                                                   scale_max=1.3,
                                                                   shift_min=-0.5,
                                                                   shift_max=0.5)

        self.pipeline = self.pipeline + AbsolutIntensityAugment(self.raw_1,
                                                                   scale_min=0.7,
                                                                   scale_max=1.3,
                                                                   shift_min=-0.5,
                                                                   shift_max=0.5)

        if upsample is not None:
            self.pipeline = self.pipeline + UpSample(self.raw_0, upsample, self.raw_0_us)
            self.pipeline = self.pipeline + UpSample(self.raw_1, upsample, self.raw_1_us)
        

        self.pipeline.setup()
        np.random.seed(os.getpid() + int(time.time()))


    def __getitem__(self, index):

        request = gp.BatchRequest()

        # request.add(self.raw_0, (1, ) + tuple(self.shape))
        # request.add(self.raw_0, (1, ) + tuple(self.shape))
        direction = random.randrange(0, self.max_direction)

        offset = offset_from_direction(direction,
                                       max_direction=self.max_direction,
                                       distance=self.distance)

        request[self.raw_0_us] = gp.Roi((0, 0, 0), (1, 256, 256))
        request[self.raw_1_us] = gp.Roi((0, ) + offset, (1, 256, 256))
        batch = self.pipeline.request_batch(request)

        out0 = batch[self.raw_0_us].data
        out1 = batch[self.raw_1_us].data

        raw_stack = np.concatenate((out0, out1), axis=0)
        shift_direction = np.array(direction, dtype=np.int64)

        return raw_stack, shift_direction

    def __len__(self):
        # return a random length
        # determined by fair dice roll ;)
        # https://xkcd.com/221/
        return 8192


class SSLDataModule(pl.LightningDataModule):

    def __init__(self, batch_size, filename, key,
                 shape=(16, 256, 256), time_min=0,
                 time_split=90, time_max=100, loader_workers=10,
                 max_direction=8, context_distance=32, upsample=(1, 4, 4)):

        super().__init__()
        self.batch_size = batch_size
        self.filename = filename
        self.key = key
        self.shape = shape
        self.time_min = time_min
        self.time_split = time_split
        self.time_max = time_max
        self.loader_workers = loader_workers
        self.max_direction = max_direction
        self.context_distance = context_distance
        self.upsample = upsample

    def setup(self, stage=None):
        self.mosaic_train = RandomShiftDataset(self.filename,
                                               self.key,
                                               time_window=(self.time_min, self.time_split),
                                               shape=self.shape,
                                               max_direction=self.max_direction,
                                               distance=self.context_distance,
                                               upsample=self.upsample)

        self.mosaic_val = RandomShiftDataset(self.filename,
                                             self.key,
                                             time_window=(self.time_split + 1, self.time_max),
                                             shape=self.shape,
                                             max_direction=self.max_direction,
                                             distance=self.context_distance,
                                             upsample=self.upsample)

    def train_dataloader(self):
        return DataLoader(self.mosaic_train, batch_size=self.batch_size, num_workers=self.loader_workers)

    def val_dataloader(self):
        return DataLoader(self.mosaic_val, batch_size=self.batch_size, num_workers=self.loader_workers)

    def test_dataloader(self):
        return None

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=False)
        try:
            parser.add_argument('--batch_size', type=int, default=8)
        except argparse.ArgumentError:
            pass
        parser.add_argument('--loader_workers', type=int, default=8)
        parser.add_argument('--filename', type=str)
        parser.add_argument('--key', type=str)
        parser.add_argument('--shape', nargs='*', default=(16, 256, 256))
        parser.add_argument('--time_min', type=int, default=0)
        parser.add_argument('--time_split', type=int, default=90)
        parser.add_argument('--time_max', type=int, default=100)
        parser.add_argument('--max_direction', type=int, default=8)
        parser.add_argument('--add_sparse_mosaic_channel', action='store_true', )
        parser.add_argument('--context_distance', type=int, default=128)

        return parser


class MosaicDataModule(pl.LightningDataModule):

    def __init__(self, batch_size, filename, key, density=None,
                 shape=(16, 256, 256), time_min=0,
                 time_split=90, time_max=100, loader_workers=10,
                 add_sparse_mosaic_channel=True, random_rot=False):
        super().__init__()
        self.batch_size = batch_size
        self.filename = filename
        self.key = key
        self.shape = shape
        self.density = density
        self.time_min = time_min
        self.time_split = time_split
        self.time_max = time_max
        self.loader_workers = loader_workers
        self.add_sparse_mosaic_channel = add_sparse_mosaic_channel
        self.random_rot = random_rot

    def setup(self, stage=None):
        self.mosaic_train = SparseChannelDataset(self.filename,
                                                 self.key,
                                                 density=self.density,
                                                 time_window=(self.time_min, self.time_split),
                                                 shape=self.shape,
                                                 add_sparse_mosaic_channel=self.add_sparse_mosaic_channel,
                                                 random_rot=self.random_rot)
        self.mosaic_val = SparseChannelDataset(self.filename,
                                               self.key,
                                               density=self.density,
                                               time_window=(self.time_split + 1, self.time_max),
                                               shape=self.shape,
                                               add_sparse_mosaic_channel=self.add_sparse_mosaic_channel,
                                               random_rot=self.random_rot)

    def train_dataloader(self):
        return DataLoader(self.mosaic_train, batch_size=self.batch_size, num_workers=self.loader_workers)

    def val_dataloader(self):
        return DataLoader(self.mosaic_val, batch_size=self.batch_size, num_workers=self.loader_workers)

    def test_dataloader(self):
        return None

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=False)
        try:
            parser.add_argument('--batch_size', type=int, default=8)
        except argparse.ArgumentError:
            pass
        parser.add_argument('--loader_workers', type=int, default=8)
        parser.add_argument('--filename', type=str)
        parser.add_argument('--key', type=str)
        parser.add_argument('--shape', nargs='*', default=(16, 256, 256))
        parser.add_argument('--time_min', type=int, default=0)
        parser.add_argument('--time_split', type=int, default=90)
        parser.add_argument('--time_max', type=int, default=100)
        parser.add_argument('--density', type=float, default=0.5)
        parser.add_argument('--add_sparse_mosaic_channel', action='store_true', )
        parser.add_argument('--random_rot', action='store_true')

        return parser



if __name__ == '__main__':
    # GunpowderDataset("/home/swolf/local/data/17-04-14/preprocessing/array.n5")

    filename = "/cephfs/swolf/swolf/bio-labels/17-04-14/raw/array.n5"
    filename = "/home/swolf/local/data/17-04-14/preprocessing/array.n5"
    filename = "/home/swolf/local/data/CTC/BF-C2DL-HSC_original/train/02_train_cropped_256.zarr"
    filename = "/home/swolf/local/data/tmp/raw.n5"
    key = "raw"

    gs = RandomShiftDataset(filename, key, shape=(128, 128), time_window=(0, 10), channels=0)

    for i in range(100):

        inp, target = gs[i]

        imsave(f"img_{i}_00.png", inp)
        imsave(f"img_{i}_01.png", target)


        # for z in range(inp.shape[1]):
        #     vecs = target[0]**2 + target[1]**2
        #     print(vecs.min(), vecs.max())
        #     img = np.stack((target[0, z], target[1, z], target[2, z]), axis=-1)
        #     imsave(f"out_{i}_{z}.png", img)
