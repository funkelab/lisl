import torch
import numpy as np
import logging
from lisl.models import MLP

logger = logging.getLogger(__name__)


class CPCLoss(torch.nn.Module):
    '''Computes a contrastive learningcontrastive predictive coding loss
    [0] https://arxiv.org/abs/1807.03748
    Args:
        stride (``int``): 
            Downsampeling factor of the embedding.
            Set stride > 1 to reduce the memory footprint
        distance (``int``):
            Distance between source and predicted pixel.
            This distance is measured after downsampling
    '''

    def __init__(self, hidden_channels, encoder_layers, distance, stride, ndim):

        super().__init__()

        self.distance = distance
        self.stride = stride

        self.prediction_cnn = MLP(hidden_channels, self.hidden_channels,
                                  n_hidden_layers=encoder_layers, n_hidden=self.hidden_channels, ndim=2)
        self.target_cnn = MLP(hidden_channels, self.hidden_channels,
                               n_hidden_layers=encoder_layers, n_hidden=self.hidden_channels, ndim=2)

    def forward(self, embedding):
        '''Compute the loss.

        Args:

            embedding (``torch.Tensor``):
                The embeddings extracted from an image patch
                Expected shape: ``(b, c, dim_1, ..., dim_d)``.

        '''

        if self.stride > 1:
            xrshift = random.randint(0, self.stride)
            yrshift = random.randint(0, self.stride)
            embedding = embedding[..., xrshift::self.stride, yrshift::self.stride]

        preds = self.prediction_cnn(embedding)
        targets = self.target_cnn(embedding)

        b, c, h, w = targets.shape
        # (b, c, h, w) -> (num_vectors, emb_dim)
        # every vector (c-dim) is a target
        targets_perm = targets.permute(0, 2, 3, 1).contiguous().reshape([-1, c])

        # select the future (south) targets to predict
        # selects all of the ones south of the current source
        preds_i = preds[:, :, :-(self.distance + 1), :] * self.embed_scale

        # (b, c, h, w) -> (b*w*h, c) (all features)
        # this ordering matches the targets
        preds_i = preds_i.permute(
            0, 2, 3, 1).contiguous().reshape([-1, c])

        # calculate the strength scores
        logits = torch.matmul(preds_i, targets_perm.transpose(-1, -2))

        # generate the target labels
        col_dim_i = h - self.distance - 1
        n = b * col_dim_i * w
        b1 = torch.arange(n) // (col_dim_i * w)
        c1 = torch.arange(n) % (col_dim_i * w)
        labels = b1 * h * w + (self.distance + 1) * w + c1
        labels = labels.to(logits.device)
        labels = labels.long()
        loss = nn.functional.cross_entropy(logits, labels)

        return loss
