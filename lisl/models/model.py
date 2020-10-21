import torch
from torch import nn
from torch.nn import functional as F


class Model(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def get_output_shape(self, input_shape):
        return self(torch.rand(*(input_shape))).data.shape


class MLP(nn.Module):
    """
    Simple MultiLayerPerceptron with Batchnorm and Dropout
    Args:
        ndim: defines the expected input dimension
              (2,3) uses 1x1 convolution layers
              dim = 0 uses linear (fully connected) layers
    """
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
