import argparse
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from lisl.pl.dataset import ShiftDataset, SparseChannelDataset, RandomShiftDataset, DSBDataset

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


class DSBDataModule(pl.LightningDataModule):

    def __init__(self, batch_size, dspath,
                 shape=(256, 256), loader_workers=10,
                 max_direction=8, context_distance=32,
                 upsample=2., patch_size_overlap_dilation=None):

        super().__init__()
        self.batch_size = batch_size
        self.dspath = dspath
        self.shape = shape
        self.loader_workers = loader_workers
        self.max_direction = max_direction
        self.context_distance = context_distance
        self.upsample = upsample
        self.patch_size_overlap_dilation = patch_size_overlap_dilation
        if self.patch_size_overlap_dilation is not None:
            # patch_size_overlap_dilation, when read from commandline might be a string
            # we cast to int, just in case
            self.patch_size_overlap_dilation = tuple(int(_) for _ in patch_size_overlap_dilation)

    def setup(self, stage=None):

        full_ds = DSBDataset(self.dspath)
        train_set_size = int(len(full_ds) * 0.9)
        val_set_size = len(full_ds) - train_set_size
        dsb_train, dsb_val = torch.utils.data.random_split(
                                full_ds, 
                                [train_set_size, val_set_size],
                                generator=torch.Generator().manual_seed(42))

        self.ds_train = ShiftDataset(
                                dsb_train,
                                self.shape,
                                self.context_distance,
                                self.max_direction,
                                self.upsample,
                                train=True,
                                return_segmentation=False,
                                patch_size_overlap_dilation=self.patch_size_overlap_dilation)

        self.ds_val = ShiftDataset(
                                dsb_val,
                                self.shape,
                                self.context_distance,
                                self.max_direction,
                                self.upsample,
                                train=False,
                                return_segmentation=True)

    def train_dataloader(self):
        return DataLoader(self.ds_train,
                          shuffle=True,
                          batch_size=self.batch_size,
                          num_workers=self.loader_workers,
                          drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.ds_val,
                          batch_size=1,
                          shuffle=False,
                          num_workers=self.loader_workers,
                          drop_last=True)

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
        parser.add_argument('--max_direction', type=int, default=8)
        # parser.add_argument('--add_sparse_mosaic_channel', action='store_true', )
        parser.add_argument('--context_distance', type=int, default=128)
        parser.add_argument('--patch_size_overlap_dilation', nargs='*', default=None)

        return parser