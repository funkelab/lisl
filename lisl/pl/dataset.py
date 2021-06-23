from torch.utils.data.dataset import Dataset, ConcatDataset
from torch.utils.data import DataLoader
from lisl.pl.utils import UpSample, offset_from_direction
from lisl.pl.utils import AbsolutIntensityAugment
import gunpowder as gp
import daisy
import zarr
import numpy as np
import random
from skimage.io import imsave, imread
import math
import pytorch_lightning as pl
import argparse
import stardist
import time
import logging
import os
from tifffile import imread as tiffread
from lisl.pl.utils import Patchify, random_offset, QuantileNormalize, Scale
import torch
import tifffile
import h5py
from PIL import Image
from torchvision.utils import make_grid
from random import choice
import zipfile
from io import BytesIO
from os.path import isfile, join

from inferno.io.transform import Compose, Transform
from inferno.io.transform.image import (RandomRotate, ElasticTransform,
  AdditiveGaussianNoise, RandomTranspose, RandomFlip, 
  FineRandomRotations, RandomGammaCorrection, RandomCrop, CenterCrop)

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
                 split="train",
                 **kwargs):

        path = Path(pathname)

        # download DSB datasetffrom stardist repo
        download_and_extract_zip_file(
            url       = 'https://github.com/mpicbg-csbd/stardist/releases/download/0.1.0/dsb2018.zip',
            targetdir = str(path/'data'),
            verbose   = 1,
        )

        # read dataset to memory
        self.X = sorted(path.glob(f'data/dsb2018/{split}/images/*.tif'))
        self.Y = sorted(path.glob(f'data/dsb2018/{split}/masks/*.tif'))
        assert all(Path(x).name==Path(y).name for x,y in zip(self.X,self.Y))
        self.X = [tiffread(str(e)) for e in self.X]
        self.Y = [tiffread(str(e)) for e in self.Y]

    def __getitem__(self, index):
        return self.X[index][None], self.Y[index]

    def __len__(self):
        return len(self.X)

class UsiigaciDataset(Dataset):

    def __init__(self,
                 pathname,
                 split="train",
                 **kwargs):

        path = Path(pathname)

        # read dataset to memory
        self.X = sorted(path.glob(f'{split}/images/*.tif'))
        self.Y = sorted(path.glob(f'{split}/masks/*.png'))
        assert all(Path(x).name[-6:-4]==Path(y).name[-6:-4] for x,y in zip(self.X,self.Y))
        self.X = [tiffread(str(e)) for e in self.X]
        self.Y = [np.array(imread(str(e))) for e in self.Y]

    def __getitem__(self, index):
        return self.X[index][None], self.Y[index]

    def __len__(self):
        return len(self.X)


class Bbbc010Dataset(Dataset):

    def __init__(self,
                 pathname,
                 split="train",
                 **kwargs):

        path = Path(pathname)

        # read dataset to memory
        self.X = sorted(path.glob(f'{split}/images/*.tif'))
        self.Y = sorted(path.glob(f'{split}/masks/*.png'))
        assert all(Path(x).name[-7:-4]==Path(y).name[-7:-4] for x,y in zip(self.X,self.Y))
        self.X = [tiffread(str(e)) for e in self.X]
        self.Y = [np.array(imread(str(e))) for e in self.Y]

    def __getitem__(self, index):
        return self.X[index][None], self.Y[index]

    def __len__(self):
        return len(self.X)


class LargeDataset(Dataset):

    def __init__(self, root, max_trials=5000):
        self.root = root
        self.max_trials = max_trials

    def sample_from_zip(self, filepath):
        zippedImgs = zipfile.ZipFile(filepath)
        file_in_zip = choice(zippedImgs.namelist())
        data = zippedImgs.read(file_in_zip)
        dataEnc = BytesIO(data)
        return file_in_zip, dataEnc

    def ignore(self, filename):
        if filename.startswith("__MACOS"):
            return True
        elif filename.endswith("ini"):
            return True
        elif "_GT/" in filename:
            # ignore CTC groundtruth files
            return True
        elif "/masks/" in filename:
            # ignore CTC groundtruth files
            return True
        elif filename.endswith("ics"):
            return True
        elif filename.endswith("mp4"):
            return True
        elif filename.endswith(".DS_Store"):
            return True
        elif filename.endswith("tar.gz"):
            return True
        elif filename.endswith(".zip"):
            return True
        elif filename.endswith(".mat"):
            return True
        elif filename.endswith(".db"):
            return True
        elif filename.endswith(".txt"):
            return True
        elif filename.endswith("/"):
            return True
        else:
            return False


    def open_image(self, filename, data):
        if self.ignore(filename):
            return None
        elif filename[-4:] in [".tif", "tiff", ".TIF", "TIFF"]:
            try:
                img = tifffile.imread(data)
            except:
                return None
            return img
        else:
            try:
                with Image.open(data) as im:
                    im = np.array(im)
                return im
            except:
                return None

    def sample_image(self):
        for trial in range(self.max_trials):
            folder = choice(os.listdir(self.root))
            file = choice(os.listdir(join(self.root, folder)))
            img_file = join(self.root, folder, file)
            if img_file.endswith("zip"):
                file_in_zip, data = self.sample_from_zip(img_file)
                img = self.open_image(file_in_zip, data)
                if img is not None:
                    return img
            elif img_file.endswith("h5"):
                try:
                    with h5py.File(img_file, "r") as inarray:
                        key = choice([k for k in inarray.keys() if "raw" in k])
                        scale = choice(['s0', 's1', 's2'])
                        img = np.array(inarray[key][scale][:])
                    return img
                except:
                    pass
            elif img_file.endswith("tar.gz"):
                # TODO read tar files
                pass
            else:
                # unknown file
                pass

        raise ValueError(f'Could not load any images after {self.max_trials} trials')


    def slice_to_2d(self, img):
        if img.ndim == 2:
            return img
        else:
            # find axes with smalles dimension
            shape = img.shape
            a = shape.index(min(shape))
            index = np.random.randint(0, shape[a])
            # take a random slice along minimum axis
            return self.slice_to_2d(img.take(index, axis=a))

    def __len__(self):
        return 10000

    def __getitem__(self, _):
        img = self.sample_image()
        img = self.slice_to_2d(img)
        # return dummy label image
        return img[None], 0*img

class ShiftDataset(Dataset):

    def __init__(self,
                 dataset,
                 shape,
                 distance,
                 max_direction,
                 train=True,
                 return_segmentation=True):

        self.root_dataset = dataset
        self.distance = distance
        self.max_direction = max_direction
        self.shape = tuple(int(_) for _ in shape)
        self.train = train
        self.return_segmentation = return_segmentation

        self.train_transforms = self.get_transforms()

    def __len__(self):
        return len(self.root_dataset)

    def get_transforms(self):
        global_transforms = Compose(RandomRotate(),
                             RandomTranspose(),
                             RandomFlip(),
                             # RandomGammaCorrection(),
                             ElasticTransform(alpha=2000., sigma=50.),)

        return global_transforms

    def __getitem__(self, index):

        x,y = self.root_dataset[index]
        y = y.astype(np.double)

        if self.train:
            x, y = self.train_transforms(x, y)

        # # reflection padding to make all shifts viable
        # x = np.pad(x, self.distance, mode='reflect')
        # y = np.pad(y, self.distance, mode='constant', constant_values=-1)
        xoff, yoff = random_offset(self.distance)

        direction = np.array([xoff, yoff]).astype(np.float32)

        x1 = x[self.distance:self.distance+self.shape[0], 
               self.distance:self.distance+self.shape[1]]
        x2 = x[self.distance+xoff:self.distance+self.shape[0]+xoff, 
               self.distance+yoff:self.distance+self.shape[1]+yoff]

        y1 = y[self.distance:self.distance+self.shape[0], 
               self.distance:self.distance+self.shape[1]]
        y2 = y[self.distance+xoff:self.distance+self.shape[0]+xoff, 
               self.distance+yoff:self.distance+self.shape[1]+yoff]

        x = np.stack((x1, x2), axis=0).astype(np.float32)
        y = np.stack((y1, y2), axis=0).astype(np.float32)

        if self.return_segmentation:
            return x, direction, y
        else:
            return x, direction


class DSBTrainAugmentations(Dataset):

    def __init__(self,
                 dataset,
                 scale=1.,
                 output_shape=(256, 256)):
      self.root_dataset = dataset
      self.scale = scale
      self.output_shape = output_shape
      self._train_transforms = None

    @property
    def train_transforms(self):
        if self._train_transforms is None:
            self._train_transforms = Compose(RandomRotate(),
                                     RandomTranspose(),
                                     RandomFlip(),
                                     QuantileNormalize(apply_to=[0]),
                                     Scale(self.scale),
                                     # RandomGammaCorrection(),
                                     ElasticTransform(alpha=2000., sigma=50.),
                                     RandomCrop(self.output_shape))

        return self._train_transforms

    def __len__(self):
        return len(self.root_dataset)

    def __getitem__(self, index):

      x, y = self.root_dataset[index]
      y = y.astype(np.double)

      x, y = self.train_transforms(x, y)
      return x, y


class DSBTestAugmentations(Dataset):

    def __init__(self,
                 dataset,
                 scale=1.,
                 output_shape=None):
        self.root_dataset = dataset
        self.scale = scale
        self.output_shape = output_shape
        self._test_transforms = None

    @property
    def test_transforms(self):
        if self._test_transforms is None:
            self._test_transforms = Compose(QuantileNormalize(apply_to=[0]),
                                            Scale(self.scale))
            if self.output_shape is not None:
                self._test_transforms.add(CenterCrop(self.output_shape))
                
        return self._test_transforms

    def __len__(self):
        return len(self.root_dataset)

    def __getitem__(self, index):

        x, y = self.root_dataset[index]
        y = y.astype(np.double)
        x, y = self.test_transforms(x, y)
        return x, y


class PatchedDataset(Dataset):

    def __init__(self,
                 dataset,
                 output_shape,
                 patch_size,
                 patch_overlap,
                 positive_radius,
                 augment=True,
                 add_negative_samples=False,
                 return_segmentation=True):
        """
        max_imbalance_dist: int
            a patch on the edge of the image has an imabalanced
            set of neighbours. We compute the distance to the averge
            neigbour (zero for patches in the center)
            If the average vector is longer than 
            max_imbalance_dist the connection is removed
        """

        self.root_dataset = dataset
        self.augment = augment
        self.return_segmentation = return_segmentation
        self.output_shape = tuple(int(_) for _ in output_shape)
        self.positive_radius = positive_radius

        self.add_negative_samples = add_negative_samples
        if self.add_negative_samples:
            self.neutral_radius = self.positive_radius * 2.
            self.negative_radius = self.positive_radius * 2.236
  


        self.patchify = Patchify(patch_size=patch_size,
                                 overlap_size=patch_overlap,
                                 dilation=1)

        # assert(patch_overlap > patch_size - patch_overlap)

        self._connection_matrix = None
        self._coords = None
        self._mask = None
        self._current_img_shape = None

    @property
    def coords(self):
        return self._coords

    def get_connections_and_coords(self, img_shape):
        if self._current_img_shape != img_shape:
            self._connection_matrix, self._coords, self._mask = \
                self.compute_connections_and_coordinates(img_shape)
        return self._connection_matrix, self._coords, self._mask

    def compute_connections_and_coordinates(self, img_shape):

        self._current_img_shape = img_shape
        x = np.arange(img_shape[-1], dtype=np.float32)
        y = np.arange(img_shape[-2], dtype=np.float32)

        coords = np.meshgrid(x, y, copy=True)
        coords = np.stack(coords, axis=0)

        patch_coords = self.patchify(torch.from_numpy(coords))
        # patch_coords.shape = (num_patches, 2, patch_width, patch_height)
        # we need the center coordinate (mean coordinate per patch)
        patch_coords = patch_coords.mean(dim=(-2, -1))
        # patch_coords.shape = (num_patches, 2)

        # Compute patch distances
        squared_distances = ((patch_coords[None] - patch_coords[:, None])**2).sum(axis=-1)
        # squared_distances.shape = (num_patches, num_patches)

        # turn squared_distances into connection matrix by thresholding

        if self.add_negative_samples:
            squared_distances[squared_distances > self.negative_radius**2] = 0
            squared_distances[squared_distances > self.neutral_radius**2] = -1
        squared_distances[squared_distances > self.positive_radius**2] = 0.
        squared_distances[squared_distances > 0.] = 1
        # remove "self" connections
        connection_matrix = squared_distances
        connection_matrix.fill_diagonal_(0)
        connection_matrix = connection_matrix.char()
        assert(torch.any(connection_matrix > 0))

        # filter non balanced patches
        avg_diff = ((connection_matrix[..., None] != 0).numpy() * (patch_coords[None] - patch_coords[:, None]).numpy())
        # avg_diff.shape = (num_patches, num_patches)
        averaged_neighbour_dist = ((avg_diff).mean(axis=0)**2).sum(axis=-1)
        # averaged_neighbour_dist.shape = (num_patches)
        # remove imbalanced neighbours

        # mask = self.max_imbalance_dist**2 < averaged_neighbour_dist
        mask = averaged_neighbour_dist != 0.

        # connection_matrix.shape = (num_patches, num_patches)
        # connection_matrix[i,j] == 1 iff patch i and j are closer than max_dist
        # but not i == j
        return connection_matrix, patch_coords, mask

    def unpack(self, sample):
      if isinstance(sample, tuple):
        if len(sample) == 2:
          x, y = sample
        else:
          x = sample[0]
          y = 0.
      else:
          x = sample
          y = 0.

      if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)

      return x, y

    def apply_patch_augmentation(self, x,
            min_alpha=0.8,
            max_alpha=1.2,
            min_beta=-0.2,
            max_beta=+0.2,
            eta_scale=0.02):

        sample_size = (x.shape[0], ) + (1, ) * (len(x.shape) - 1)

        alpha = np.random.uniform(min_alpha, max_alpha, size=sample_size)
        beta = np.random.uniform(min_beta, max_beta, size=sample_size)
        noise_level = np.random.uniform(0, eta_scale)
        eta = noise_level * np.random.randn(*x.shape)

        return (alpha * x + beta + eta).astype(np.float32)

    def __len__(self):
        return len(self.root_dataset)

    def __getitem__(self, index):

        x,y = self.unpack(self.root_dataset[index])

        conn_mat, abs_coords, mask = self.get_connections_and_coords(x.shape)
        patches = self.patchify(x)

        if self.augment:
            patches = self.apply_patch_augmentation(patches.numpy())

        if self.return_segmentation:
            return x, patches, abs_coords, conn_mat, mask, y
        else:
            return x, patches, abs_coords, conn_mat, mask

class Threeclass(Transform):
    """Convert segmentation to 3 class"""

    def __init__(self, inner_distance=4):
        super().__init__()
        self.inner_distance = inner_distance

    def tensor_function(self, gt):
        gt_stardist = stardist.geometry.star_dist(gt, n_rays=8)
        background = gt == 0
        inner = gt_stardist.min(axis=-1) > self.inner_distance
        # classes 0: boundaries, 1: inner_cell, 2: background
        threeclass = (2 * background) + inner
        return threeclass.astype(np.long)

class ZarrEmbeddingDataset(Dataset):
    """
    Attributes
    ----------
    ds_file: str
        File to zarr array that contains dataset
    """

    def __init__(self,
                 ds_file,
                 emb_key,
                 crop_to=(256, 256)):
        super().__init__()
        self.ds_file = ds_file
        self.ds_data = zarr.open(ds_file, "r")
        self.emb_key = emb_key
        self.threeclass = Threeclass
        self.rcrop = RandomCrop(crop_to)

    def __len__(self):
        if not hasattr(self, "_length"):
            self._length = len(self.ds_data.keys())
        return self._length

    def __getitem__(self, idx):

        while f"{idx}/raw" not in self.ds_data:
            print(f"Can not find raw[{idx}] in {self.ds_file}")
            idx = (idx + 1) % self._length

        raw = self.ds_data[f"{idx}/raw"][:][None].astype(np.float32)
        emb_in = self.ds_data[f"{idx}/{self.emb_key}"]
        if emb_in.ndim == 2:
            embedding = emb_in[:][None].astype(np.float32)
        elif emb_in.ndim == 4:
            embedding = emb_in[0].astype(np.float32)
        else:
            embedding = emb_in[:].astype(np.float32)

        gt = self.ds_data[f"{idx}/gt"][:]
        tc = self.threeclass(inner_distance=2)(gt)
        raw, embedding, tc, gt = self.rcrop(raw, embedding, tc, gt)

        return raw, embedding, tc, gt

class AugmentedZarrEmbeddingDataset(Dataset):
    def __init__(self,
                ds_file_prefix,
                ds_file_postfix,
                augmentations,
                emb_key,
                crop_to=(256, 256)):
    
        ds_list = []
        dsf_files = [f"{ds_file_prefix}{i:02d}{ds_file_postfix}" for i in range(augmentations)]

        # filter non existent datasets
        dsf_files = [f for f in dsf_files if os.path.exists(f)]

        ds_list = [ZarrEmbeddingDataset(dsf,
                                        emb_key=emb_key,
                                        crop_to=crop_to) for dsf in dsf_files]
        self._dataset = ConcatDataset(ds_list)

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        return self._dataset[idx]

if __name__ == '__main__':

    ds = AugmentedZarrEmbeddingDataset(ds_file_prefix="/nrs/funke/wolfs2/lisl/datasets/fast_dsb_coord_aug_",
                                       ds_file_postfix=".zarr",
                                       emb_key="interm_cooc_emb",
                                       augmentations=20)

    
    idx = 1
    from tqdm import tqdm

    for idx in tqdm(range(0, 1000)):
        batch = ds[idx]
        raw, embedding, tc, target = batch
        print(idx, np.unique(target))
        tc_vis = np.stack([tc==0, tc==1, tc==2], axis=-1).astype(np.float32)
        imsave(f"/nrs/funke/wolfs2/lisl/experiments/dev00/{idx:06}_tc_vis.png", tc_vis)
        imsave(f"/nrs/funke/wolfs2/lisl/experiments/dev00/{idx:06}_raw.png", raw[0])
        target_vis = target.astype(np.float32)
        imsave(f"/nrs/funke/wolfs2/lisl/experiments/dev00/{idx:06}_target_vis.png", target_vis)

    # grid = make_grid(y.detach().cpu().softmax(1)).permute(1, 2, 0).numpy()
