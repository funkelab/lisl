from confnets.models.unet import UNet, UNet3d
from confnets.layers import Identity
from torch import nn
from torch.nn import functional as F
import torch
import numpy as np
from partialconv.models.partialconv2d import PartialConv2d
from partialconv.models.partialconv3d import PartialConv3d
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models

import numpy as np
import scipy.sparse as sparse
from lisl.pl.utils import Patchify
import math

class UnPatchify(object):
    """
    Rearranges patchified tensor into a tiled image with spatial dimenstions.
    """

    def __init__(self, patch_size, overlap_size):
        self.patch_size = patch_size
        self.patch_stride = self.patch_size - overlap_size
        assert(self.patch_stride >= 0)

    def __call__(self, x):
        x = x.unsqueeze(0)
        b, c, h, w = x.size()

        # patch up the images
        # (b, c, h, w) -> (b, c*patch_size, L)
        x = F.unfold(x, kernel_size=self.patch_size, stride=self.patch_stride)

        # (b, c*patch_size, L) -> (b, nb_patches, width, height)
        x = x.transpose(2, 1).contiguous().view(
            b, -1, self.patch_size, self.patch_size)

        # reshape to have (b x patches, c, h, w)
        x = x.view(-1, c, self.patch_size, self.patch_size)
        x = x.squeeze(0)
        x = x.unsqueeze(0)
        print(x.size())
        b, c, h, w = x.size()
        x = x.squeeze(0)
        return x

class PatchedResnet(nn.Module):

    def __init__(
            self,
            in_channels,
            out_channels,
            resnet_size=50,
            pretrained=False):

        super().__init__()

        if resnet_size == 18:
            model = models.resnet18(pretrained=pretrained)
            features_in_last_layer = 512
        elif resnet_size == 50:
            model = models.resnet50(pretrained=pretrained)
            features_in_last_layer = 2048
        else:
            raise NotImplementedError()

        if in_channels != 3:
            ptweights = model.conv1.weight.mean(dim=1, keepdim=True).data
            model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, padding=3, bias=False)
            if in_channels == 1:
                model.conv1.weight.data = ptweights

        layers = list(model.children())[:-1]
        model = torch.nn.Sequential(*layers)
        self.resnet = model.train()

        # Commonly used Non-linear projection head
        # see https://arxiv.org/pdf/1906.00910.pdf
        # or https://arxiv.org/pdf/2002.05709.pdf
        self.head = torch.nn.Sequential(
            nn.Linear(features_in_last_layer, features_in_last_layer),
            nn.ReLU(),
            nn.Linear(features_in_last_layer, out_channels))

        self.add_spatial_dim = False

    def forward(self, patches):
        # patches has to be a Tensor with
        # patch.shape == (minibatch, channels, patch_width, patch_height)

        h = self.resnet(patches)
        h = torch.flatten(h, 1)
        # get output directly after avg pooling layer
        # h.shape == (minibatch, features_in_last_layer)

        # apply small MLP
        z = self.head(h)
        # z.shape = (minibatch, outchannels)
        if self.add_spatial_dim:
            return h[..., None, None], z[..., None, None]

        return h, z

def get_unet_kernels(ndim):
    if ndim == 3:
        dsf = ((1, 3, 3), (1, 2, 2), (1, 2, 2))
        ks = (
            ((1, 3, 3), (3, 3, 3)),
            ((1, 3, 3), (3, 3, 3)),
            ((1, 3, 3), (1, 3, 3)),
            ((1, 3, 3), (1, 3, 3)),
        )
    elif ndim == 2:
        dsf = ((3, 3), (2, 2), (2, 2))
        ks = (((3, 3), (3, 3)),
              ((3, 3), (3, 3)),
              ((3, 3), (3, 3)),
              ((3, 3), (3, 3)))
    else:
        raise NotImplementedError()

    return dsf, ks


class MLP(nn.Module):
    def __init__(self, n_input, n_classes, n_hidden=512, n_hidden_layers=1, p=0.1, ndim=2):
        super().__init__()

        self.n_input = n_input
        self.n_classes = n_classes
        self.n_hidden = n_hidden
        self.n_hidden_layers = n_hidden_layers

        if n_hidden is None:
            n_hidden = []

        if ndim == 3:
            conv = nn.Conv3d
            bn = nn.BatchNorm3d
            dout = nn.Dropout3d
        elif ndim == 2:
            conv = nn.Conv2d
            bn = nn.BatchNorm2d
            dout = nn.Dropout2d
        elif ndim == 0:
            conv = nn.Linear
            bn = nn.BatchNorm1d
            dout = nn.Dropout

        n_feats = n_input

        if isinstance(n_hidden_layers, dict):
            n_hidden_layers = 4
            n_classes = 1
        print(n_hidden_layers)

        self.block_forward = nn.Sequential()
        for i in range(n_hidden_layers):
            self.block_forward.add_module(f"drop_{i}", dout(p=p))
            if ndim > 0:
                self.block_forward.add_module(f"conv_{i}", conv(n_feats, n_hidden, 1, bias=False))
            else:
                self.block_forward.add_module(f"conv_{i}", conv(n_feats, n_hidden, bias=False))
            self.block_forward.add_module(f"bn_{i}", bn(n_hidden))
            self.block_forward.add_module(f"rl_{i}", nn.ReLU(inplace=True))
            n_feats = n_hidden

        self.block_forward.add_module("drop_fin", dout(p=p))
        if ndim > 0:
            self.block_forward.add_module("conv_fin", conv(n_feats, n_classes, 1, bias=True))
        else:
            self.block_forward.add_module("conv_fin", conv(n_feats, n_classes, bias=True))

    def forward(self, x):
        if self.n_hidden_layers == -1:
            return x[:, :self.n_classes]
        else:
            return self.block_forward(x)


class PartialConvLayer(nn.Module):

    def __init__(self, f_in, f_out, kernel_size=3, return_mask=True, activation=F.relu, stride=1, dim=3, norm="batch"):
        super().__init__()

        if dim == 3:
            convtype = PartialConv3d
            if norm == "instance":
                self.norm = nn.InstanceNorm3d(f_in)
            elif norm == "batch":
                self.norm = nn.BatchNorm3d(f_in)
            else:
                self.norm = Identity()

        elif dim == 2:
            convtype = PartialConv2d

            if norm == "instance":
                self.norm = nn.InstanceNorm2d(f_in)
            elif norm == "batch":
                self.norm = nn.BatchNorm2d(f_in)
            else:
                self.norm = Identity()

        self.conv = convtype(f_in, f_out,
                             kernel_size=kernel_size,
                             padding=(kernel_size - 1) // 2,
                             stride=stride,
                             return_mask=return_mask,
                             multi_channel=True)

        self.activation = activation

    def forward(self, input):
        if isinstance(input, (list, tuple)):
            inp, mask = input
        else:
            mid = input.shape[1] // 2
            inp = input[:, :mid]
            mask = input[:, mid:]

        inp = self.norm(inp)
        out, mask = self.conv(inp, mask_in=mask)

        if self.activation:
            out = self.activation(out)

        return out, mask


class PartialMergeLayer(nn.Module):

    def forward(self, A, B):
        if isinstance(A, (list, tuple)):
            inpA, maskA = A
        else:
            midA = A.shape[1] // 2
            inpA = A[:, :midA]
            maskA = A[:, midA:]

        if isinstance(B, (list, tuple)):
            inpB, maskB = B
        else:
            midB = B.shape[1] // 2
            inpB = B[:, :midB]
            maskB = B[:, midB:]

        return torch.cat((inpA, inpB), dim=1), torch.cat((maskA, maskB), dim=1)


class PartialMaxPooling(nn.Module):

    def __init__(self, scale_factor, dim=3):
        super().__init__()
        if dim == 3:
            self.mp = nn.MaxPool3d(kernel_size=scale_factor,
                                   stride=scale_factor,
                                   padding=0)
        else:
            self.mp = nn.MaxPool2d(kernel_size=scale_factor,
                                   stride=scale_factor,
                                   padding=0)

    def forward(self, input):

        if isinstance(input, (list, tuple)):
            inp, mask = input
        else:
            mid = input.shape[1] // 2
            inp = input[:, :-mid]
            mask = input[:, -mid:]

        return self.mp(inp), self.mp(mask)


class Upsample(nn.Module):
    def __init__(self, scale_factor=2, mode='nearest'):
        super(Upsample, self).__init__()
        self.mode = mode
        self.scale_factor = scale_factor

    def forward(self, input):
        return nn.functional.interpolate(input, scale_factor=self.scale_factor, mode=self.mode)


class PartialUpsample(nn.Module):

    def __init__(self, scale_factor, mode):
        super().__init__()
        self.us = Upsample(scale_factor=scale_factor, mode=mode)

    def forward(self, input):
        if isinstance(input, (list, tuple)):
            inp, mask = input
        else:
            mid = input.shape[1] // 2
            inp = input[:, :-mid]
            mask = input[:, -mid:]

        return self.us(inp), self.us(mask)


class PartialStripMask(nn.Module):

    def forward(self, input):
        if isinstance(input, (list, tuple)):
            inp, mask = input
        else:
            mid = input.shape[1] // 2
            inp = input[:, :-mid]

        return inp


class PartialConvUNet2D(UNet):

    def __init__(self, **super_kwargs):
        super().__init__(dim=2, **super_kwargs)

    def construct_input_module(self):
        return Identity()

    def encoder_kernel_size(self, depth):
        if depth == 0:
            return 7
        elif depth == 1:
            return 5
        else:
            return 3

    def construct_encoder_module(self, depth):
        return Identity()

    def construct_decoder_module(self, depth):
        if depth > 0:
            f_in = self.fmaps[depth - 1] + self.fmaps[depth]
        else:
            f_in = self.in_channels + self.fmaps[0]
        f_intermediate = self.fmaps[depth]
        # do not reduce to f_out yet - this is done in the output module
        f_out = f_intermediate if depth == 0 else self.fmaps[depth - 1]
        kernel_size = 3
        return PartialConvLayer(f_in,
                                f_out,
                                kernel_size=kernel_size,
                                activation=nn.LeakyReLU(0.2, True),
                                norm="batch",
                                dim=2)

    def construct_layer(self, f_in, f_out, kernel_size=3):
        return PartialConvLayer(f_in, f_out, kernel_size=kernel_size, dim=3)

    def construct_downsampling_module(self, depth):

        f_in = self.in_channels if depth == 0 else self.fmaps[depth - 1]
        f_out = self.fmaps[depth]
        kernel_size = self.encoder_kernel_size(depth)
        scale_factor = self.scale_factors[depth]
        return PartialConvLayer(f_in,
                                f_out,
                                kernel_size=kernel_size,
                                activation=F.relu,
                                norm="batch",
                                stride=scale_factor,
                                dim=2)

    def construct_upsampling_module(self, depth):

        scale_factor = self.scale_factors[depth]
        if scale_factor[0] == 1:
            assert scale_factor[1] == scale_factor[2]
        return PartialUpsample(scale_factor, mode=self.upsampling_mode)

    def construct_merge_module(self, depth):
        return PartialMergeLayer()

    def construct_output_module(self):
        return nn.Sequential(
            PartialConvLayer(self.fmaps[0],
                             self.out_channels,
                             kernel_size=3,
                             activation=None,
                             norm=None,
                             stride=1,
                             dim=2),
            PartialStripMask())

    def construct_base_module(self):
        return Identity()


class PartialConvUNet3D(UNet):

    def __init__(self, **super_kwargs):
        super().__init__(dim=3, **super_kwargs)

    def construct_layer(self, f_in, f_out, kernel_size=3):
        return PartialConvLayer(f_in, f_out, kernel_size=kernel_size, dim=3)

    def construct_downsampling_module(self, depth):
        scale_factor = self.scale_factors[depth]
        return PartialMaxPooling(scale_factor, dim=3)

    def construct_upsampling_module(self, depth):
        scale_factor = self.scale_factors[depth]
        if scale_factor[0] == 1:
            assert scale_factor[1] == scale_factor[2]
        return PartialUpsample(scale_factor, mode="nearest")

    def construct_merge_module(self, depth):
        return PartialMergeLayer()

    def construct_output_module(self):
        return nn.Sequential(
            PartialConvLayer(self.fmaps[0],
                             self.out_channels,
                             kernel_size=3,
                             activation=None,
                             norm=None,
                             stride=1,
                             dim=3),
            PartialStripMask())

    def init_submodules(self):
        # run a forward pass to initialize all submodules (with in_channels=nn.INIT_DELAYED)
        with torch.no_grad():
            inp = torch.zeros((2, 2 * self.in_channels, *self.divisibility_constraint), dtype=torch.float32)
            self.forward(inp)
