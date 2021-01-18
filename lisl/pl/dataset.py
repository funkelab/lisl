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
from tifffile import imread as tiffread
from skimage.transform import rescale
from lisl.pl.utils import Patchify, random_offset
import torch

from inferno.io.transform import Compose
from inferno.io.transform.image import (RandomRotate, ElasticTransform,
  AdditiveGaussianNoise, RandomTranspose, RandomFlip, FineRandomRotations)

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

    def __init__(self,
                 filename,
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
        self.upsample = upsample
        self.time_window = time_window

        self.pipeline = self.build_source()

        self.pipeline = self.pipeline + gp.RandomLocation() + IntensityDiffFilter(self.raw_0, min_distance=0.1, channel=0)

        # add  augmentations
        self.pipeline = self.pipeline + gp.ElasticAugment([40, 40],
                                                          [2, 2],
                                                          [0, math.pi / 2.0],
                                                          prob_slip=-1,
                                                          spatial_dims=2)

        self.pipeline = self.pipeline + AbsolutIntensityAugment(self.raw_0,
                                                                   scale_min=0.9,
                                                                   scale_max=1.1,
                                                                   shift_min=-0.1,
                                                                   shift_max=0.1)

        self.pipeline = self.pipeline + AbsolutIntensityAugment(self.raw_1,
                                                                   scale_min=0.9,
                                                                   scale_max=1.1,
                                                                   shift_min=-0.1,
                                                                   shift_max=0.1)

        if upsample is not None:
            self.pipeline = self.pipeline + UpSample(self.raw_0, upsample, self.raw_0_us)
            self.pipeline = self.pipeline + UpSample(self.raw_1, upsample, self.raw_1_us)
        

        self.pipeline.setup()
        np.random.seed(os.getpid() + int(time.time()))

    def build_source(self):
        data = daisy.open_ds(filename, key)

        if self.time_window is None:
            source_roi = gp.Roi(data.roi.get_offset(), data.roi.get_shape())
        else:
            offs = list(data.roi.get_offset())
            offs[1] += self.time_window[0]
            sh = list(data.roi.get_shape())
            offs[1] = self.time_window[1] - self.time_window[0]
            source_roi = gp.Roi(tuple(offs), tuple(sh))

        voxel_size = gp.Coordinate(data.voxel_size)

        return gp.ZarrSource(filename,
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
                             })


    def __getitem__(self, index):

        request = gp.BatchRequest()

        # request.add(self.raw_0, (1, ) + tuple(self.shape))
        # request.add(self.raw_0, (1, ) + tuple(self.shape))
        direction = random.randrange(0, self.max_direction)

        offset = offset_from_direction(direction,
                                       max_direction=self.max_direction,
                                       distance=self.distance)
        if self.upsample is not None:
            ref0, ref1 = self.raw_0_us, self.raw_1_us
        else:
            ref0, ref1 = self.raw_0, self.raw_1

        request[ref0] = gp.Roi((0, 0, 0), (1, 256, 256))
        request[ref1] = gp.Roi((0, ) + offset, (1, 256, 256))
            
        batch = self.pipeline.request_batch(request)

        out0 = batch[ref0].data
        out1 = batch[ref1].data

        raw_stack = np.concatenate((out0, out1), axis=0)
        shift_direction = np.array(direction, dtype=np.int64)

        return raw_stack, shift_direction

    def __len__(self):
        # return a random length
        # determined by fair dice roll ;)
        # https://xkcd.com/221/
        return 8192

from csbdeep.utils import download_and_extract_zip_file
from pathlib import Path

class DSBDataset(Dataset):

    def __init__(self,
                 pathname,
                 **kwargs):

        path = Path(pathname)

        # download DSB datasetffrom stardist repo
        download_and_extract_zip_file(
            url       = 'https://github.com/mpicbg-csbd/stardist/releases/download/0.1.0/dsb2018.zip',
            targetdir = str(path/'data'),
            verbose   = 1,
        )

        # read dataset to memory
        self.X = sorted(path.glob('data/dsb2018/train/images/*.tif'))
        self.Y = sorted(path.glob('data/dsb2018/train/masks/*.tif'))
        assert all(Path(x).name==Path(y).name for x,y in zip(self.X,self.Y))
        self.X = [tiffread(str(e)) for e in self.X]
        self.Y = [tiffread(str(e)) for e in self.Y]
        print(self.X[0].shape)
        print(self.Y[0].shape)

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

    def __len__(self):
        return len(self.X)

def augment(x,
            min_alpha=0.8,
            max_alpha=1.2,
            min_beta=-0.2,
            max_beta=+0.2,
            eta_scale=0.05):

    alpha = random.uniform(min_alpha, max_alpha)
    beta = random.uniform(min_beta, max_beta)
    noise_level = random.uniform(0, eta_scale)
    eta = noise_level * np.random.randn(*x.shape)

    return alpha * x + beta + eta

class ShiftDataset(Dataset):

    def __init__(self,
                 dataset,
                 shape,
                 distance,
                 max_direction,
                 max_scale=2.,
                 train=True,
                 patch_size_overlap_dilation=None,
                 return_segmentation=True):

        self.root_dataset = dataset
        self.distance = distance
        self.max_direction = max_direction
        self.max_scale = max_scale
        self.shape = tuple(int(_) for _ in shape)
        self.train = train
        self.return_segmentation = return_segmentation

        self.train_transforms, self.fine_transforms = self.get_transforms()

        if patch_size_overlap_dilation is not None:
          self.patchify = Patchify(patch_size=patch_size_overlap_dilation[0],
                                   overlap_size=patch_size_overlap_dilation[1],
                                   dilation=patch_size_overlap_dilation[2])
        else:
          self.patchify = None

    def __len__(self):
        return len(self.root_dataset)

    def get_transforms(self):
        global_transforms = Compose(RandomRotate(),
                             RandomTranspose(),
                             RandomFlip(),
                             ElasticTransform(alpha=2000., sigma=50.),
                             AdditiveGaussianNoise(sigma=0.05))

        fine_transforms = ElasticTransform(alpha=2000., sigma=50.)

        return global_transforms, fine_transforms

    def __getitem__(self, index):

        x,y = self.root_dataset[index]
        y = y.astype(np.double)

        if self.train:
            factor = 1. + self.max_scale * np.random.rand()

            print(x.shape, y.shape)

            x = rescale(x, factor, order=1)
            y = rescale(y, factor, order=0)

            print(x.shape, y.shape)
            x, y = self.train_transforms(x, y)

        # reflection padding to make all shifts viable
        x = np.pad(x, self.distance, mode='reflect')
        y = np.pad(y, self.distance, mode='constant', constant_values=-1)

        if self.max_direction > 0:
          direction = random.randrange(0, self.max_direction)
          xoff, yoff = offset_from_direction(direction,
                                             max_direction=self.max_direction,
                                             distance=self.distance)
        else:
          direction = 0
          xoff, yoff = random_offset(self.distance)

        direction = np.array([direction, xoff, yoff]).astype(np.float32)



        x1 = x[self.distance:self.distance+self.shape[0], 
               self.distance:self.distance+self.shape[1]]
        x2 = x[self.distance+xoff:self.distance+self.shape[0]+xoff, 
               self.distance+yoff:self.distance+self.shape[1]+yoff]

        y1 = y[self.distance:self.distance+self.shape[0], 
               self.distance:self.distance+self.shape[1]]
        y2 = y[self.distance+xoff:self.distance+self.shape[0]+xoff, 
               self.distance+yoff:self.distance+self.shape[1]+yoff]
        
        if self.train:
            x1 = self.fine_transforms(x1)
            x2 = self.fine_transforms(x2)
            x1 = augment(x1)
            x2 = augment(x2)

        x = np.stack((x1, x2), axis=0).astype(np.float32)
        y = np.stack((y1, y2), axis=0).astype(np.float32)

        if self.patchify is not None:
          x = self.patchify(torch.from_numpy(x))

        if self.return_segmentation:
            return x, direction, y
        else:
            return x, direction

if __name__ == '__main__':
    # GunpowderDataset("/home/swolf/local/data/17-04-14/preprocessing/array.n5")

    filename = "/cephfs/swolf/swolf/bio-labels/17-04-14/raw/array.n5"
    filename = "/home/swolf/local/data/17-04-14/preprocessing/array.n5"
    filename = "/home/swolf/local/data/CTC/BF-C2DL-HSC_original/train/02_train_cropped_256.zarr"
    filename = "/home/swolf/local/data/tmp/raw.n5"
    key = "raw"

    # filename = "~/mnt/cephfs/swolf/data/dsb2018/train"
    filename = Path.home()/"tmp"
    distance = 10

    gs = ShiftDataset(DSBDataset(filename),
                      (256, 256),
                      distance,
                      8,
                      train=True,
                      return_segmentation=False)

    for i in range(10):

        inp, direction = gs[i]

        print("inp", inp.shape)
        # print("target", target.shape)

        for c in range(inp.shape[0]):
            imsave(f"img_{i}_{c}_s.png", inp[c])
        # imsave(f"img_{i}_0_t.png", target[0])

        # for z in range(inp.shape[1]):
        #     vecs = target[0]**2 + target[1]**2
        #     print(vecs.min(), vecs.max())
        #     img = np.stack((target[0, z], target[1, z], target[2, z]), axis=-1)
        #     imsave(f"out_{i}_{z}.png", img)
