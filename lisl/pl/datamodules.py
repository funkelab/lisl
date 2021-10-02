import argparse
import pytorch_lightning as pl
import torch
import os
from torch.utils.data import DataLoader
from lisl.pl.dataset import (PatchedDataset, SparseChannelDataset, 
                             RandomShiftDataset, DSBDataset, UsiigaciDataset, Bbbc010Dataset,
                             DSBTrainAugmentations, DSBTestAugmentations,
                             LargeDataset, AugmentedZarrEmbeddingDataset,
                             AugmentedZarrDataset, ZarrEmbeddingDataset,
                             LiveCellDataset, TissueNetDataset)
from torchvision import transforms, datasets
from lisl.pl.utils import QuantileNormalizeTorchTransform

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
        print(self.filename)
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


class AnchorDataModule(pl.LightningDataModule):

    def __init__(self, batch_size, dspath,
                 shape=(256, 256), loader_workers=10, positive_radius=32,
                 image_scale=1., patch_size=16, patch_overlap=5):

        super().__init__()
        self.batch_size = batch_size
        self.dspath = dspath
        self.shape = tuple(int(_) for _ in shape)
        self.loader_workers = loader_workers
        self.patch_size = patch_size
        self.patch_overlap = patch_overlap
        self.positive_radius = positive_radius
        self.scale = image_scale

    def setup_datasets(self):
      raise NotImplementedError()

    def setup(self, stage=None):
      
        dsb_train, dsb_val = self.setup_datasets()

        self.ds_train = PatchedDataset(
                                dsb_train,
                                self.shape,
                                self.patch_size,
                                self.patch_overlap,
                                self.positive_radius,
                                augment=True,
                                return_segmentation=False)

        self.ds_val = PatchedDataset(
                                dsb_val,
                                self.shape,
                                self.patch_size,
                                self.patch_overlap,
                                self.positive_radius,
                                augment=False,
                                return_segmentation=True)

    def train_dataloader(self):
        return DataLoader(self.ds_train,
                          shuffle=True,
                          batch_size=self.batch_size,
                          num_workers=self.loader_workers,
                        #   worker_init_fn=pl.utilities.seed.seed_everything,
                          drop_last=False)

    def val_dataloader(self):
        return DataLoader(self.ds_val,
                          batch_size=1,
                          shuffle=False,
                          num_workers=2,
                        #   worker_init_fn=pl.utilities.seed.seed_everything,
                          drop_last=False)

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
        parser.add_argument('--dspath', type=str)
        parser.add_argument('--shape', nargs='*', default=(256, 256))
        parser.add_argument('--patch_size', type=int, default=16)
        parser.add_argument('--patch_overlap', type=int, default=5)
        parser.add_argument('--positive_radius', type=int, default=64)
        parser.add_argument('--image_scale', type=float, default=1.)
        parser.add_argument('--imgfolder', type=str)
        parser.add_argument('--train_annotations', type=str)
        parser.add_argument('--val_annotations', type=str)

        return parser
        
class LargeDataModule(AnchorDataModule):

  def setup_datasets(self):
    full_ds = LargeDataset(self.dspath)
    dsb_val = DSBDataset('/nrs/funke/wolfs2/lisl/datasets/dsb')

    full_ds = DSBTrainAugmentations(full_ds,
                                    scale=self.scale,
                                    output_shape=self.shape)
    dsb_val = DSBTestAugmentations(dsb_val,
                                   scale=self.scale,
                                   output_shape=(256, 256))

    return full_ds, dsb_val

class DSBDataModule(AnchorDataModule):

  def setup_datasets(self):

    full_ds = DSBDataset(self.dspath)
    train_set_size = int(len(full_ds) * 0.90)
    val_set_size = len(full_ds) - train_set_size
    torch.manual_seed(100)
    dsb_train, dsb_val = torch.utils.data.random_split(
                            full_ds,
                            [train_set_size, val_set_size])

    # add different augmentations to train and val datasets
    dsb_train = DSBTrainAugmentations(dsb_train,
                                      scale=self.scale,
                                      output_shape=self.shape)
    dsb_val = DSBTestAugmentations(dsb_val,
                                   scale=self.scale,
                                   output_shape=(256, 256))

    return dsb_train, dsb_val


class UsiigaciDataModule(AnchorDataModule):

  def setup_datasets(self):

    ds_train = UsiigaciDataset(
        self.dspath,
        "train")

    ds_val = UsiigaciDataset(
        self.dspath,
        "val")

    # add different augmentations to train and val datasets
    ds_train = DSBTrainAugmentations(ds_train,
                                     scale=self.scale,
                                     output_shape=self.shape)
    ds_val = DSBTestAugmentations(ds_val,
                                  scale=self.scale,
                                  output_shape=(256, 256))

    return ds_train, ds_val

class Bbbc010DataModule(AnchorDataModule):

  def setup_datasets(self):

    ds_train = Bbbc010Dataset(
        self.dspath,
        "train")

    ds_val = Bbbc010Dataset(
        self.dspath,
        "val")

    # add different augmentations to train and val datasets
    ds_train = DSBTrainAugmentations(ds_train,
                                     scale=self.scale,
                                     output_shape=self.shape)
    ds_val = DSBTestAugmentations(ds_val,
                                  scale=self.scale,
                                  output_shape=(256, 256))

    return ds_train, ds_val

class OpenImagesDataModule(AnchorDataModule):

    def setup_datasets(self):

      normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

      full_ds = datasets.ImageFolder(
          self.dspath,
          transforms.Compose([
            transforms.RandomResizedCrop(self.shape),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
          ]))

      train_set_size = int(len(full_ds) * 0.90)
      val_set_size = len(full_ds) - train_set_size
      torch.manual_seed(100)
      dsb_train, dsb_val = torch.utils.data.random_split(
                                full_ds,
                                [train_set_size, val_set_size])
        
      return dsb_train, dsb_val

class CelebADataModule(AnchorDataModule):

    def setup_datasets(self):

      dsb_train = datasets.CelebA(self.dspath,
                                  download=False,
                                  split='train',
                                  transform=transforms.Compose([
                                    transforms.Resize(self.shape),
                                    transforms.CenterCrop(self.shape),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                         std=[0.5, 0.5, 0.5])
                              ]))

      dsb_val = datasets.CelebA(self.dspath,
                                  download=False,
                                  transform=transforms.Compose([
                                    transforms.Resize(self.shape),
                                    transforms.CenterCrop(self.shape),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                         std=[0.5, 0.5, 0.5])
                              ]))

      return dsb_train, dsb_val

class LiveCellDataModule(AnchorDataModule):

    def __init__(self,
                 batch_size, dspath,
                 image_folder=None,
                 annotation_file=None,
                 val_annotation_file=None,
                 shape=(256, 256), loader_workers=10, positive_radius=32,
                 image_scale=1., patch_size=16, patch_overlap=5):
                 
        super().__init__(batch_size,
                         dspath,
                         shape=shape,
                         loader_workers=loader_workers,
                         positive_radius=positive_radius,
                         image_scale=image_scale,
                         patch_size=patch_size,
                         patch_overlap=patch_overlap)

        self.image_folder = image_folder
        self.annotation_file = annotation_file
        self.val_annotation_file = val_annotation_file


    def setup_datasets(self):   
        train_ds = LiveCellDataset(self.image_folder,
                                    self.annotation_file,
                                    crop_to=self.shape)

        val_ds = LiveCellDataset(self.image_folder,
                                 self.val_annotation_file,
                                 crop_to=(256, 256),
                                 augment=False)

        return train_ds, val_ds

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = super(LiveCellDataModule, LiveCellDataModule).add_model_specific_args(parent_parser)
        parser.add_argument('--image_folder', type=str)
        parser.add_argument('--annotation_file', type=str)
        parser.add_argument('--val_annotation_file', type=str)

        return parser

class TissueNetDataModule(AnchorDataModule):

    def setup_datasets(self):   
        train_ds = TissueNetDataset(os.path.join(self.dspath, "tissuenet_v1.0_train.npz"),
                                    crop_to=self.shape)

        val_ds = TissueNetDataset(os.path.join(self.dspath, "tissuenet_v1.0_val.npz"),
                                 crop_to=(256, 256),
                                 augment=False)

        return train_ds, val_ds

class TissueNetThreeClassDataModule(pl.LightningDataModule):

    def __init__(self, batch_size, dspath, loader_workers, 
                 target_transform="threeclass", ds_limit=None, crop_to=(256, 256)):
        self.dspath = dspath 
        self.loader_workers = loader_workers
        self.batch_size = batch_size
        self.target_transform = target_transform
        self.ds_limit = ds_limit
        self.crop_to = crop_to

    def setup(self, stage=None):
        self.train = TissueNetDataset(os.path.join(self.dspath, "tissuenet_v1.0_train.npz"),
                                      target_transform=self.target_transform,
                                      limit=self.ds_limit,
                                      crop_to=self.crop_to)

        self.val = TissueNetDataset(os.path.join(self.dspath, "tissuenet_v1.0_val.npz"),
                                     target_transform=self.target_transform,
                                     augment=False)

        self.test = TissueNetDataset(os.path.join(self.dspath, "tissuenet_v1.0_test.npz"),
                                     target_transform=self.target_transform,
                                     augment=False)

    def train_dataloader(self):
        return DataLoader(self.train,
                          batch_size=self.batch_size,
                          num_workers=self.loader_workers,
                          shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.val,
                          batch_size=1,
                          num_workers=1,
                          shuffle=False) 

    def test_dataloader(self):
        return DataLoader(self.test,
                          batch_size=1,
                          num_workers=1,
                          shuffle=False)


    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=False)
        try:
            parser.add_argument('--batch_size', type=int, default=8)
        except argparse.ArgumentError:
            pass

        parser.add_argument('--dspath', type=str, required=True)
        parser.add_argument('--loader_workers', type=int, default=8)
        parser.add_argument('--ds_limit', default=None, type=int, nargs='+')
        return parser

class PrecomputedThreeClassDataModule(pl.LightningDataModule):

    def __init__(self, batch_size, ds_file_prefix,
                 ds_file_postfix, augmentations,
                 emb_keys, loader_workers, val_ds_file,
                 test_ds_file, target_transform="threeclass", 
                 crop_to=(256, 256), ds_limit=None, min_spatial_div=None):

        super().__init__()
        self.batch_size = batch_size
        self.ds_file_prefix = ds_file_prefix
        self.ds_file_postfix = ds_file_postfix
        self.augmentations = augmentations
        self.min_spatial_div = min_spatial_div
        self.emb_keys = emb_keys
        self.target_transform = target_transform
        self.crop_to = crop_to
        self.loader_workers = loader_workers

        self.val_ds_file = val_ds_file
        self.test_ds_file = test_ds_file
        self.limit = ds_limit

    def setup(self, stage=None):
        self.train = AugmentedZarrEmbeddingDataset(self.ds_file_prefix,
                                                   self.ds_file_postfix,
                                                   self.augmentations,
                                                   self.emb_keys,
                                                   target_transform=self.target_transform,
                                                   crop_to=self.crop_to,
                                                   limit=self.limit,
                                                   min_spatial_div=self.min_spatial_div)

        self.test = ZarrEmbeddingDataset(self.test_ds_file,
                                        self.emb_keys,
                                        target_transform=self.target_transform,
                                        crop_to=None,
                                        min_spatial_div=self.min_spatial_div,
                                        p_dropout=0)

    def train_dataloader(self):
        return DataLoader(self.train,
                          batch_size=self.batch_size,
                          num_workers=self.loader_workers,
                          shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.test,
                          batch_size=1,
                          num_workers=1,
                          shuffle=False) 

    def test_dataloader(self):
        return DataLoader(self.test,
                          batch_size=1,
                          num_workers=1,
                          shuffle=False)


    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=False)
        try:
            parser.add_argument('--batch_size', type=int, default=8)
        except argparse.ArgumentError:
            pass

        parser.add_argument('--loader_workers', type=int, default=8)
        parser.add_argument('--ds_file_prefix', type=str, required=True)
        parser.add_argument('--ds_file_postfix', type=str, required=True)
        parser.add_argument('--augmentations', type=int, required=True)
        parser.add_argument('--emb_keys', type=str, required=True, nargs='+')
        parser.add_argument('--target_transform', type=str, default="threeclass")
        parser.add_argument('--crop_to', default=(256, 256))
        parser.add_argument('--min_spatial_div', default=16)
        parser.add_argument('--ds_limit', default=None, type=int, nargs='+')
        parser.add_argument('--val_ds_file', type=str)
        parser.add_argument('--test_ds_file', type=str, required=True)

        return parser


class ThreeClassDataModule(pl.LightningDataModule):

    def __init__(self, batch_size, ds_file,
                 loader_workers, val_ds_file,
                 test_ds_file, target_transform="threeclass", 
                 crop_to=(256, 256), ds_limit=None):

        super().__init__()
        self.batch_size = batch_size
        self.ds_file = ds_file
        self.target_transform = target_transform
        self.crop_to = crop_to
        self.loader_workers = loader_workers

        self.val_ds_file = val_ds_file
        self.test_ds_file = test_ds_file
        self.limit = ds_limit

    def setup(self, stage=None):
        self.train = AugmentedZarrDataset(self.ds_file,
                                          crop_to=self.crop_to,
                                          limit=self.limit)

        self.test = AugmentedZarrDataset(self.test_ds_file,
                                        crop_to=None,
                                        augment=False)

    def train_dataloader(self):
        return DataLoader(self.train,
                          batch_size=self.batch_size,
                          num_workers=self.loader_workers,
                          shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.test,
                          batch_size=1,
                          num_workers=1,
                          shuffle=False) 

    def test_dataloader(self):
        return DataLoader(self.test,
                          batch_size=1,
                          num_workers=1,
                          shuffle=False)


    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=False)
        try:
            parser.add_argument('--batch_size', type=int, default=8)
        except argparse.ArgumentError:
            pass

        parser.add_argument('--loader_workers', type=int, default=8)
        parser.add_argument('--ds_file', type=str, required=True)
        parser.add_argument('--target_transform', type=str, default="threeclass")
        parser.add_argument('--crop_to', default=(256, 256))
        parser.add_argument('--min_spatial_div', default=16)
        parser.add_argument('--ds_limit', default=None, type=int, nargs='+')
        parser.add_argument('--val_ds_file', type=str)
        parser.add_argument('--test_ds_file', type=str, required=True)

        return parser


# class LiveCellDataModule(pl.LightningDataModule):

#     def __init__(self, batch_size, image_folder,
#                  annotation_file, loader_workers, val_annotation_file,
#                  test_annotation_file, target_transform=None, 
#                  crop_to=(256, 256), ds_limit=None):

#         super().__init__()
#         self.batch_size = batch_size
#         self.image_folder = image_folder
#         self.annotation_file = annotation_file
#         self.target_transform = target_transform
#         self.crop_to = crop_to
#         self.loader_workers = loader_workers

#         self.test_annotation_file = test_annotation_file
#         self.limit = ds_limit

#     def setup(self, stage=None):


#     def train_dataloader(self):
#         return DataLoader(self.train,
#                           batch_size=self.batch_size,
#                           num_workers=self.loader_workers,
#                           shuffle=True)
    
#     def val_dataloader(self):
#         return None 

#     def test_dataloader(self):
#         return DataLoader(self.test,
#                           batch_size=1,
#                           num_workers=1,
#                           shuffle=False)


#     @staticmethod
#     def add_model_specific_args(parent_parser):
#         parser = argparse.ArgumentParser(
#             parents=[parent_parser], add_help=False)
#         try:
#             parser.add_argument('--batch_size', type=int, default=8)
#         except argparse.ArgumentError:
#             pass

#         parser.add_argument('--loader_workers', type=int, default=8)
#         parser.add_argument('--image_folder', type=str, required=True)
#         parser.add_argument('--annotation_file', type=str, required=True)
#         parser.add_argument('--val_annotation_file', type=str, required=False, default=None)
#         parser.add_argument('--test_annotation_file', type=str, required=True)
#         parser.add_argument('--target_transform', type=str, default=None)
#         parser.add_argument('--crop_to', default=(256, 256))
#         parser.add_argument('--ds_limit', default=None, type=int, nargs='+')

#         return parser