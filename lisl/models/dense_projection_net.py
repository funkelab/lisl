from .model import Model
from funlib.learn.torch.models.conv4d import Conv4d
import torch


class DenseProjectionNet(Model):

    def __init__(self, base_encoder, h_channels, out_channels):

        super().__init__()

        self.base_encoder = base_encoder
        self.in_channels = base_encoder.out_channels
        self.h_channels = h_channels
        self.out_channels = out_channels
        self.dims = base_encoder.dims
        self.in_shape = base_encoder.in_shape
        self.out_shape = base_encoder.out_shape

        conv = {
            2: torch.nn.Conv2d,
            3: torch.nn.Conv3d,
            4: Conv4d
        }[self.dims]

        self.projection_head = torch.nn.Sequential(
            conv(self.in_channels, h_channels, (1,)*self.dims),
            torch.nn.ReLU(),
            conv(h_channels, out_channels, (1,)*self.dims)
        )

    def forward(self, raw_0, raw_1=None):

        # (b, c, dim_1, ..., dim_d)
        h_0 = self.base_encoder(raw_0)
        z_0 = self.projection_head(h_0)
        z_0_norm = torch.nn.functional.normalize(z_0)

        if raw_1 is not None:

            h_1 = self.base_encoder(raw_1)
            z_1 = self.projection_head(h_1)
            z_1_norm = torch.nn.functional.normalize(z_1)

            return h_0, h_1, z_0_norm, z_1_norm

        return h_0, z_0_norm

