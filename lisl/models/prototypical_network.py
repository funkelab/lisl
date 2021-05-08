import torch
from torch import nn
from torch.nn import functional as F
from .model import MLP
from pytorchcv.models.erfnet import ERFNet

class PrototypicalNetwork(nn.Module):
    def __init__(self, in_channels, inst_out_channels, n_sem_classes, hidden_size=512):
        super().__init__()
        self.in_channels = in_channels
        self.inst_out_channels = inst_out_channels
        self.hidden_size = hidden_size
        self.n_sem_classes = n_sem_classes
        self.ndim = 2
        self.spatial_instance_encoder = MLP(in_channels,
                                            inst_out_channels,
                                            n_hidden=hidden_size,
                                            n_hidden_layers=4,
                                            p=0.1,
                                            ndim=0)
       
        self.semantic_encoder = MLP(in_channels,
                                    self.n_sem_classes,
                                    n_hidden=hidden_size,
                                    n_hidden_layers=4,
                                    p=0.1,
                                    ndim=0)


    def forward(self, inputs):
        abs_coords = inputs[..., :self.ndim]
        z = inputs[..., self.ndim:]
        semantic_embeddings = self.semantic_encoder(z)
        spatial_instance_embeddings = self.spatial_instance_encoder(z)
        spatial_instance_embeddings[..., :self.ndim] += abs_coords
        return spatial_instance_embeddings, semantic_embeddings


class ERFNetsmooth(ERFNet):
    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.head = nn.Sequential(
            torch.nn.Upsample(scale_factor=2,
                              mode="nearest"),
            nn.Conv2d(in_channels=self.head.in_channels,
                      out_channels=self.head.in_channels,
                      kernel_size=5,
                      padding=2,
                      bias=True),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.head.in_channels,
                      out_channels=self.head.in_channels,
                      kernel_size=3,
                      padding=1,
                      bias=True),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.head.in_channels,
                      out_channels=self.head.in_channels,
                      kernel_size=3,
                      padding=1,
                      bias=True),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.head.in_channels,
                      out_channels=self.head.out_channels,
                      kernel_size=3,
                      padding=1,
                      bias=True))


class PrototypicalERFNetwork(nn.Module):
    def __init__(self, in_channels, inst_out_channels, n_sem_classes, hidden_size=512, in_size=(256, 256)):
        super().__init__()
        self.in_channels = in_channels
        self.inst_out_channels = inst_out_channels
        self.hidden_size = hidden_size
        self.n_sem_classes = n_sem_classes
        self.ndim = 2

        # ERFNet default parameters
        downs = [1, 1, 1, 0, 0]
        channels = [16, 64, 128, 64, 16]
        dilations = [[1], [1, 1, 1, 1, 1, 1], [
        1, 2, 4, 8, 16, 2, 4, 8, 16], [1, 1, 1], [1, 1, 1]]
        dropout_rates = [[0.0], [0.03, 0.03, 0.03, 0.03, 0.03, 0.03], [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3],
                    [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]

        # reduce the number of input dimensions to fit the ERF default parameters
        # the model requires channels[0] >  in_channels
        self.init_layer = torch.nn.Sequential(
            nn.Conv2d(in_channels, 8, 1),
            torch.nn.ReLU())

        self.encoder = ERFNetsmooth(channels,
                                    dilations,
                                    dropout_rates,
                                    downs,
                                    in_channels=8,
                                    in_size=in_size,
                                    num_classes=inst_out_channels + self.n_sem_classes)

        self.coords = None

    def forward(self, inputs):

        inputs[:, 0] -= inputs[:, 0].mean(dim=(1, 2), keepdims=True)
        inputs[:, 1] -= inputs[:, 1].mean(dim=(1, 2), keepdims=True)

        z_init = self.init_layer(inputs)
        embeddings = self.encoder(z_init)
        # TODO: update pytorch to use tensor_split
        # inst, alpha, sem = torch.tensor_split(embeddings, (inst_out_channels, 1, self.n_sem_classes), dim=1)
        inst = embeddings[:, :self.inst_out_channels]
        sem = embeddings[:, self.inst_out_channels:]

        inst[:, 0] += inputs[:, 1]
        inst[:, 1] += inputs[:, 0]

        inst = self.add_coords(inst)

        return inst, sem

    def add_coords(self, spatial_instance_embeddings):
        sie = spatial_instance_embeddings
        if self.coords is None or self.coords.shape[-2:] != sie.shape[-2:]:
            with torch.no_grad():
                cx = torch.arange(sie.shape[-2], dtype=sie.dtype)
                cy = torch.arange(sie.shape[-1], dtype=sie.dtype)
                self.coords = torch.meshgrid(cx, cy)
                self.coords = torch.stack(self.coords, axis=0)[None]

        self.coords = self.coords.to(spatial_instance_embeddings.device)

        spatial_instance_embeddings[:, :self.ndim] += self.coords
        return spatial_instance_embeddings

