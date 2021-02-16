from torch.nn.modules.module import Module
import torch
from torch import Tensor
from torch.nn import functional as F
import gc
from time import sleep

class AnchorLoss(Module):
    r"""

    Args:
        anchor_radius (float): The attraction of anchor points is non linear (sigmoid function).
            The used nonlinear distance is sigmoid(euclidean_distance - anchor_radius)
            Therefore the anchors experience a stronger attraction when their distance is 
            smaller than the anchor_radius
    """

    def __init__(self, anchor_radius) -> None:
        super().__init__()
        self.anchor_radius = anchor_radius

    def forward(self, embedding, abs_coords, patch_mask=None) -> Tensor:

        abs_embedding = embedding + abs_coords

        coord_diff = abs_coords[:, :, None] - abs_coords[:, None]
        e0 = embedding[:, :, None]
        e1 = embedding[:, None]

        # compute all pairwise distances of anchor embeddings
        diff = (e0 - e1) + coord_diff
        dist = diff.norm(2, dim=-1)
        # dist.shape = (b, p, p)

        nonlinear_dist = (dist - self.anchor_radius).sigmoid()
        
        # only matched patches (e.g. patches close in proximity)
        # contripute to the loss
        if patch_mask is not None:
            nonlinear_dist = nonlinear_dist[patch_mask]

        return nonlinear_dist.sum()

class AnchorPlusLoss(Module):
    r"""

    Args:
        anchor_radius (float): The attraction of anchor points is non linear (sigmoid function).
            The used nonlinear distance is sigmoid(euclidean_distance - anchor_radius)
            Therefore the anchors experience a stronger attraction when their distance is 
            smaller than the anchor_radius
        spatial_dim (int): Number of spatial dimensions
    """

    def __init__(self, anchor_radius,) -> None:
        super().__init__()
        self.anchor_radius = anchor_radius
        self.xnt_loss = torch.nn.CrossEntropyLoss(reduction='sum')
        # self.proxy_anchor_loss = losses.ProxyAnchorLoss()

    def set_anchor_radius(self, new_radius):
        self.anchor_radius = new_radius

    def multiply_anchor_radius(self, gamma):
        if self.anchor_radius > 1.:
            self.anchor_radius *= gamma

    def forward(self, embedding, abs_coords, patch_mask=None) -> Tensor:
        # embedding.shape = (b, p, c)
        # b = batchsize
        # p = number of patches
        # c = number of embedding

        sdim = abs_coords.shape[-1]

        abs_embedding = embedding[..., :sdim] + abs_coords

        coord_diff = abs_coords[:, :, None] - abs_coords[:, None]
        e0 = embedding[:, :, None]
        e1 = embedding[:, None, :]

        # compute all pairwise distances of anchor embeddings
        diff = (e0 - e1)
        diff[..., :2] += coord_diff
        dist = diff.norm(2, dim=-1)

        sim = 1-(((dist*5)/self.anchor_radius)-5).sigmoid()

        patch_mask = patch_mask
        neg_patch_mask = patch_mask.clone()
        for b in range(len(neg_patch_mask)):
            neg_patch_mask[b].fill_diagonal_(True)
        
        loss = 0.
        neg_masked_sim = sim.masked_fill(neg_patch_mask, float('-inf'))

        target = torch.zeros(neg_patch_mask.shape[-1], dtype=torch.long, device=sim.device)
        tmp_smpl = []

        for row_index in range(patch_mask.shape[-1]):

            column_mask = patch_mask[:, :, row_index]
            # sim[column_mask].masked_fill(~patch_mask[column_mask], float('-inf'))

            positive_sample = sim[column_mask, row_index]
            negative_samples = neg_masked_sim[column_mask]
            samples = torch.cat((positive_sample[..., None], negative_samples), dim=-1)
            
            tmp_smpl.append((samples))
            loss = loss + self.xnt_loss(samples, target[:len(samples)])

        return loss

