from .model import Model
from funlib.learn.torch.models.conv4d import Conv4d
import funlib
import torch

class UNet(Model):

    def __init__(
            self,
            in_channels,
            in_shape,
            num_fmaps,
            fmap_inc_factors,
            downsample_factors,
            kernel_size_down,
            kernel_size_up,
            padding,
            constant_upsample):

        super().__init__()


        self.unet = funlib.learn.torch.models.UNet(
            in_channels=in_channels,
            num_fmaps=num_fmaps,
            fmap_inc_factors=fmap_inc_factors,
            downsample_factors=downsample_factors,
            kernel_size_down=kernel_size_down,
            kernel_size_up=kernel_size_up,
            # padding=padding,
            constant_upsample=constant_upsample)

        self.dims = len(downsample_factors[0])
        self.in_shape = in_shape
        self.out_shape = self.get_output_shape([1, 1, *self.in_shape])[2:]
        self.out_channels = num_fmaps

    def forward(self, x):

        return self.unet(x)
