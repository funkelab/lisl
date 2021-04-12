import torch
from torch import nn
from torch.nn import functional as F
from .model import MLP

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
