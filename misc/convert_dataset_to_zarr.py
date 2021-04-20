import zarr
import sys
from torch.utils.data.dataset import Dataset
import numpy as np
from lisl.pl.dataset import DSBDataset
from lisl.pl.utils import Patchify, random_offset, QuantileNormalize, Scale
import torch

from inferno.io.transform import Compose
from inferno.io.transform.image import CenterCrop
from tqdm import tqdm

def write_data_to_zarr(z_array, ):

    first_batch.dtype()

class PreprocessingDataset(Dataset):

    def __init__(self,
                 dataset,
                 scale=1.,
                 output_shape=(256, 256)):
      self.root_dataset = dataset
      self.scale = scale
      self.output_shape = output_shape
      self._transform = None

    @property
    def transform(self):
        if self._transform is None:
            self._transform = Compose(QuantileNormalize(apply_to=[0]),
                                      Scale(self.scale))
            if self.output_shape is not None:
                self._transform.add(CenterCrop(self.output_shape))

        return self._transform

    def __len__(self):
        return len(self.root_dataset)

    def __getitem__(self, index):

      x, y = self.root_dataset[index]
      y = y.astype(np.double)

      x, y = self.transform(x, y)
      return x, y



def convert_dataset_to_zarr(dataset, output_file, mode="a", prefix_key=""):

    z_array = zarr.open(output_file, mode=mode)
    
    first_batch = dataset[0]
    z_array = zarr.open(output_file, mode=mode)

    # check that dataset returns image and segmentation
    assert(len(first_batch) == 2)

    for idx, (x, y) in tqdm(enumerate(dataset)):
        x_out = z_array.create_dataset(f"{prefix_key}/raw/{idx}",
                           data=x[0],
                           overwrite=True)

        y_out = z_array.create_dataset(f"{prefix_key}/gt_segmentation/{idx}",
                           data=y,
                           overwrite=True)


if __name__ == '__main__':
    
    for split in ["train", "test"]:

        output_shape = None

        ds = PreprocessingDataset(DSBDataset("/tmp", split=split),
                                scale=1.,
                                output_shape=output_shape)

        # write all images (full epoch) of a datset into a zarr array
        convert_dataset_to_zarr(ds, sys.argv[1], prefix_key=split, mode="a")
