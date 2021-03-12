from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from lisl.pl.utils import UpSample, offset_from_direction
from lisl.pl.utils import AbsolutIntensityAugment
import gunpowder as gp
import daisy
import numpy as np
import random
from skimage.io import imsave, imread
import math
import pytorch_lightning as pl
import argparse
import time
import logging
import os
from tifffile import imread as tiffread
from lisl.pl.utils import Patchify, random_offset, QuantileNormalize, Scale
import torch

from csbdeep.utils import download_and_extract_zip_file
from pathlib import Path
from inferno.io.transform import Compose
from inferno.io.transform.image import (RandomRotate, ElasticTransform,
  AdditiveGaussianNoise, RandomTranspose, RandomFlip, 
  FineRandomRotations, RandomGammaCorrection, RandomCrop, CenterCrop)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


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

class PrintDebug(gp.BatchFilter):

    def prepare(self, request):
        print("prepare", request)

    def process(self, batch, request):
        print("process", request)
        print("process", batch)

# class GpDataset(Dataset):

#     def __init__(self,
#                  filename,
#                  key,
#                  shape,
#                  voxel_size,
#                  elastic_kwargs=None,
#                  min_intensity=None):

#         self.filename = filename
#         self.key = key
#         self.shape = shape
#         self.raw_0 = gp.ArrayKey('RAW_0')
#         self.voxel_size = voxel_size
#         self.pipeline = self.build_source()
#         self.pipeline += gp.Crop(self.raw_0,
#                                  roi=gp.Roi((250, 1500, 700, 1000), (20, 1500, 712, 612)))
#         self.pipeline += gp.RandomLocation()
#         # self.pipeline += PrintDebug()
#         if min_intensity is not None:
#             self.pipeline += IntensityFilter(self.raw_0, 0,
#                                              min_intensity=min_intensity,
#                                              quantile=0.9,
#                                              rejection_sampling=True)
        
#         self.pipeline += gp.Normalize(self.raw_0)

#         # # add  augmentations
#         # if elastic_kwargs is not None:
#         #     self.pipeline += gp.ElasticAugment(**elastic_kwargs)
#         # self.pipeline += gp.IntensityAugment(self.raw_0,
#         #                                      scale_min=0.8,
#         #                                      scale_max=1.2,
#         #                                      shift_min=-0.2,
#         #                                      shift_max=0.2,
#         #                                      clip=False)
#         # self.pipeline += gp.SimpleAugment()
#         # self.pipeline += gp.NoiseAugment(self.raw_0, var=0.01, clip=False)
#         self.pipeline.setup()

#     def build_source(self):
#         return gp.ZarrSource(self.filename,
#                              {
#                                  self.raw_0: self.key
#                              },
#                              {
#                                  self.raw_0: gp.ArraySpec(
#                                      roi=gp.Roi((250, 300, 700, 1000), (20, 52, 712, 612)),
#                                      interpolatable=True,
#                                      voxel_size=self.voxel_size)
#                              })
# # t = 250ish, z=[300, 352] in voxels so [1500-1760] in "real world units", y=[700, 1312], x=[1000,1612]

#     def __getitem__(self, index):
#         request = gp.BatchRequest()
#         request[self.raw_0] = gp.Roi((0, ) * len(self.shape), self.shape)
#         batch = self.pipeline.request_batch(request)
#         out = batch[self.raw_0].data

#         return out

#     def __len__(self):
#         # return a random length
#         # determined by fair dice roll ;)
#         # https://xkcd.com/221/
#         return 8192


class GpDataset(Dataset):

    def __init__(self,
                 filename,
                 key,
                 shape,
                 voxel_size,
                 elastic_kwargs=None,
                 min_intensity=None):

        self.filename = filename
        self.key = key
        self.shape = shape
        self.raw_0 = gp.ArrayKey('RAW_0')
        self.voxel_size = voxel_size
        self.pipeline = self.build_source()
        self.pipeline += gp.Crop(self.raw_0,
                                 roi=gp.Roi((250, 1500, 700, 1000), (20, 1500, 712, 612)))
        self.pipeline += gp.RandomLocation()
        if min_intensity is not None:
            self.pipeline += IntensityFilter(self.raw_0, 0,
                                             min_intensity=min_intensity,
                                             quantile=0.9,
                                             rejection_sampling=True)

        self.pipeline += gp.Normalize(self.raw_0)

        # # add  augmentations
        # if elastic_kwargs is not None:
            # self.pipeline += gp.ElasticAugment(**elastic_kwargs)
        self.pipeline += gp.IntensityAugment(self.raw_0,
                                             scale_min=0.8,
                                             scale_max=1.2,
                                             shift_min=-0.2,
                                             shift_max=0.2,
                                             clip=False)
        self.pipeline += gp.SimpleAugment(mirror_only=[2,3],
                                          transpose_only=[2,3])
        self.pipeline += gp.NoiseAugment(self.raw_0, var=0.01, clip=True)

        self.pipeline.setup()

    def build_source(self):
        return gp.ZarrSource(self.filename,
                             {
                                 self.raw_0: self.key
                             },
                             {
                                 self.raw_0: gp.ArraySpec(
                                     interpolatable=True,
                                     voxel_size=self.voxel_size)
                             })

    def __getitem__(self, index):
        request = gp.BatchRequest()
        request[self.raw_0] = gp.Roi((0, ) * len(self.shape), self.shape)
        batch = self.pipeline.request_batch(request)
        out = batch[self.raw_0].data

        return out

    def __len__(self):
        # return a random length
        # determined by fair dice roll ;)
        # https://xkcd.com/221/
        return 8192

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
            img = batch[self.key].data[self.channel]
            print("non zero ratio", np.count_nonzero(img) / img.size)
            non_zero_ratio = np.count_nonzero(img) / img.size
            # if q > self.min_intensity:
            if non_zero_ratio > 0.9:
                is_good_batch = True
                print(np.histogram(img))
        return batch

if __name__ == '__main__':

    ds = GpDataset("/nrs/funke/malinmayorc/140521/140521.zarr",
                   "raw",
                   (2, 20, 128, 128),
                   (1, 5, 1, 1),
                   min_intensity=300)

    data = ds[0]
    print(np.histogram(data))

    for i, x in enumerate(data):
        for j, y in enumerate(x):
            print(y.min(), y.max())
            imsave(f"q/{i}_{j}.png", y / y.max())

    exit()
    # GunpowderDataset("/home/swolf/local/data/17-04-14/preprocessing/array.n5")

    filename = "/cephfs/swolf/swolf/bio-labels/17-04-14/raw/array.n5"
    filename = "/home/swolf/local/data/17-04-14/preprocessing/array.n5"
    filename = "/home/swolf/local/data/CTC/BF-C2DL-HSC_original/train/02_train_cropped_256.zarr"
    filename = "/home/swolf/local/data/tmp/raw.n5"
    key = "raw"

    # filename = "~/mnt/cephfs/swolf/data/dsb2018/train"
    filename = Path.home()/"tmp"
    distance = 10

    folder = "/home/swolf/mnt/janelia/nrs/funke/wolfs2/data/BBBC010/images"

    gs = Bbbc010Dataset(
        folder,
        "val"
        # transforms.ToTensor()
        )
    # gs = ShiftDataset(DSBDataset(filename),
    #                   (256, 256),
    #                   distance,
    #                   8,
    #                   train=True,
    #                   return_segmentation=False)

    for i in range(10):

        inp = gs[i][0]

        print("inp", np.array(inp))
        # print("target", target.shape)

        for c in range(inp.shape[0]):
            print(f"q/img_{i}_{c}_s.png")
            imsave(f"q/img_{i}_{c}_s.png", inp[c])
        # imsave(f"img_{i}_0_t.png", target[0])

        # for z in range(inp.shape[1]):
        #     vecs = target[0]**2 + target[1]**2
        #     print(vecs.min(), vecs.max())
        #     img = np.stack((target[0, z], target[1, z], target[2, z]), axis=-1)
        #     imsave(f"out_{i}_{z}.png", img)
