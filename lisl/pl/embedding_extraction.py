import numpy as np
# from pl_bolts.models.self_supervised.cpc.networks import CPCResNet101
from tiktorch.models.dunet import DUNet
from argparse import ArgumentParser
from lisl.datasets import Dataset
from lisl.predict.volume import predict_volume
from funlib.learn.torch.models import UNet
from lisl.pl.model import get_unet_kernels

class GPModels():
  def __init__(self, model, in_shape, out_shape):
    self.in_shape


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--input_dataset', '-d')
    parser.add_argument('--ds_key', '-k')
    parser.add_argument('--out_dir', '-o')
    args = parser.parse_args()

    # CPCResNet101(dummy_batch)
    # pf = Patchify(patch_size=trainer.datamodule.patch_size, overlap_size=trainer.datamodule.patch_size - 1)
    model = DUNet(1, 4, N=4, return_hypercolumns=True).eval().cuda()

    dataset = Dataset(args.input_dataset, (args.ds_key, ))

    # dsf, ks = get_unet_kernels(2)
    # model = UNet(in_channels=1,
    #              num_fmaps_out=8,
    #              num_fmaps=4,
    #              fmap_inc_factor=2,
    #              downsample_factors=dsf,
    #              kernel_size_down=ks,
    #              kernel_size_up=ks,
    #              activation="GELU",
    #              num_heads=1,
    #              constant_upsample=True)

    predict_volume(model,
                   dataset,
                   args.out_dir,
                   args.input_dataset,
                   ("embedding", ),
                   input_name='input_',
                   checkpoint=None,
                   normalize_factor='skip',
                   model_output=0,
                   in_shape=(1, 256, 256),
                   out_shape=(1, 256, 256),
                   spawn_subprocess=False,
                   num_workers=0,
                   z_is_time=False,
                   apply_voxel_size=False)


