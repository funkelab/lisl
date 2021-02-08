from torch.nn.modules.module import Module
import torch
from torch import Tensor
from torch.nn import functional as F

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

    def __init__(self, anchor_radius) -> None:
        super().__init__()
        self.anchor_radius = anchor_radius
        # self.proxy_anchor_loss = losses.ProxyAnchorLoss()


    def forward(self, embedding, abs_coords, patch_mask=None) -> Tensor:
        # embedding.shape = (b, p, c)
        # b = batchsize
        # p = number of patches
        # c = number of embedding

        sdim = abs_coords.shape[-1]

        abs_embedding = embedding[..., :sdim] + abs_coords

        coord_diff = abs_coords[:, :, None] - abs_coords[:, None]
        e0 = embedding[:, :, None, ..., :sdim]
        e1 = embedding[:, None, :, ..., :sdim]

        # compute all pairwise distances of anchor embeddings
        diff = (e0 - e1) + coord_diff
        dist = diff.norm(2, dim=-1)
        # dist.shape = (b, p, p)

        nonlinear_dist = (dist - self.anchor_radius).sigmoid()
        
        # only matched patches (e.g. patches close in proximity)
        # contripute to the loss
        if patch_mask is not None:
            nonlinear_dist = nonlinear_dist[patch_mask]

        spatial_loss = nonlinear_dist.sum()

        # NTXent loss for non spatial dimensions:
        # also known as temperature-scaled cross-entropy loss

        # max_val = torch.max(pos_pairs, torch.max(neg_pairs, dim=1, keepdim=True)[0].half()) ###This is the line change
        # numerator = torch.exp(pos_pairs - max_val).squeeze(1)
        # denominator = torch.sum(torch.exp(neg_pairs - max_val), dim=1) + numerator
        # log_exp = torch.log((numerator/denominator) + 1e-20)

        if embedding.shape[-1] == sdim:
            # only spatial dimensions found
            # returning spatial loss
            return spatial_loss

        sim = F.cosine_similarity(embedding[:, :, None, ..., sdim:],
                                  embedding[:, None, ..., sdim:], -1, 1e-8)
        # sim.shape= (b, p, p)

        positive_mask = (dist.detach() < self.anchor_radius)
        negative_mask = (~positive_mask).clone()

        if patch_mask is not None:
            positive_mask.masked_fill_(~patch_mask, 0)
            negative_mask.masked_fill_(~patch_mask, 0)
        else:
            # mask out diagonal entries
            mask = torch.eye(positive_mask.shape[-2], positive_mask.shape[-1]).bool()
            positive_mask.masked_fill_(mask[None].to(positive_mask.device), 0)

        margin = 0.1
        alpha = 32.

        num_pos_samples = positive_mask.sum(dim=(1, 2))
        num_neg_samples = negative_mask.sum(dim=(1, 2))
        print("pos/neg samples", num_pos_samples, num_neg_samples)

        pos_sim = -alpha*((sim - margin).masked_fill_(~positive_mask, 0))
        pos_loss = (F.softplus((pos_sim.logsumexp(dim=2)))).sum(1)
        pos_loss = (pos_loss / (num_pos_samples + 1e-8)).sum()

        neg_sim = +alpha*((sim + margin).masked_fill_(~negative_mask, 0))
        neg_loss = (F.softplus((pos_sim.logsumexp(dim=2)))).sum(1)
        neg_loss = (neg_loss / (num_neg_samples + 1e-8)).sum()

        return spatial_loss + pos_loss + neg_loss