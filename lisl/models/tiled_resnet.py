from .model import Model

import torch
from torchvision import datasets, transforms, models



class UNet(Model):

    def __init__(
            self,
            in_channels,
            in_shape,
            num_fmaps,
            fmap_inc_factor,
            downsample_factors,
            kernel_size_down,
            kernel_size_up,
            padding,
            constant_upsample):

        super().__init__()

        self.resnet = model = models.resnet50(pretrained=True)
        # remove avg Pooling layer
        self.resnet = torch.nn.Sequential(*(list(model.children())[:-2]))

        self.in_shape = in_shape
        self.out_shape = self.get_output_shape([1, 1, *self.in_shape])[2:]
        self.out_channels = num_fmaps

    def forward(self, x):

        return self.unet(x)
