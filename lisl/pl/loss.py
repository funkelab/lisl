from torch.nn.modules.module import Module
import torch
from torch import Tensor
from torch.nn import functional as F
import gc
from time import sleep

import matplotlib
matplotlib.use('agg')
from matplotlib import collections  as mc
import pylab
import matplotlib.pyplot as plt
import time
from skimage.io import imsave

class AnchorLoss(Module):
    r"""

    Args:
        anchor_radius (float): The attraction of anchor points is non linear (sigmoid function).
            The used nonlinear distance is sigmoid(euclidean_distance - anchor_radius)
            Therefore the anchors experience a stronger attraction when their distance is 
            smaller than the anchor_radius
    """

    def __init__(self, temperature) -> None:
        super().__init__()
        self.temperature = temperature

    def nonlinearity(self, distance):
        return 1 - (-distance.pow(2) / self.temperature).exp()

    def forward(self, embedding, abs_coords, patch_mask) -> Tensor:

        coord_diff = abs_coords[:, :, None] - abs_coords[:, None]
        e0 = embedding[:, :, None]
        e1 = embedding[:, None]

        # compute all pairwise distances of anchor embeddings
        diff = (e0 - e1) + coord_diff
        dist = diff.norm(2, dim=-1)
        # dist.shape = (b, p, p)

        nonlinear_dist = self.nonlinearity(dist)
        
        # only matched patches (e.g. patches close in proximity)
        # contripute to the loss
        nonlinear_dist = nonlinear_dist[patch_mask == 1]

        return nonlinear_dist.sum()


    def multiply_temperature(self, gamma):
        if self.temperature > 1.:
            self.temperature *= gamma
            
class LinearAnchorLoss(AnchorLoss):
    def nonlinearity(self, distance):
        return distance

class SigmoidAnchorLoss(AnchorLoss):
    def nonlinearity(self, distance):
        return (distance - self.temperature).sigmoid()


class AnchorPlusLoss(Module):
    r"""

    Args:
        anchor_radius (float): The attraction of anchor points is non linear (sigmoid function).
            The used nonlinear distance is sigmoid(euclidean_distance - anchor_radius)
            Therefore the anchors experience a stronger attraction when their distance is 
            smaller than the anchor_radius
        spatial_dim (int): Number of spatial dimensions
    """

    def __init__(self, anchor_radius, temperature) -> None:
        super().__init__()
        self.temperature = temperature
        self.anchor_radius = anchor_radius
        self.xnt_loss = torch.nn.CrossEntropyLoss(reduction='sum')
        # self.proxy_anchor_loss = losses.ProxyAnchorLoss()

    def set_temperature(self, new_temp):
        self.temperature = new_temp

    def multiply_temperature(self, gamma):
        if self.temperature > 1.:
            self.temperature *= gamma

    def similarity(self, distance):
        return (1/(distance)).clamp_(max=1/self.anchor_radius)

    def anchor_loss(self, sim, patch_mask):

        loss = 0
        num_patches = sim.shape[-1]

        # copy sim array and mask all values that are not negative pairs with -inf
        neg_masked_sim = sim.masked_fill(patch_mask != -1, float('-inf'))


        # create mask of all columns that contain positive or negative pairs
        training_mask = (patch_mask<0).any(dim=2) * (patch_mask>0).any(dim=2)
        # fold column and batch dimension
        training_mask = training_mask.view(-1)

        # remove all columns that have no positive or negative pairs
        sim = sim.view(-1, num_patches)[training_mask]
        neg_masked_sim = neg_masked_sim.view(-1, num_patches)[training_mask]

        patch_mask = patch_mask.view(-1, num_patches)[training_mask]
        loss = 0.
        target = torch.zeros(patch_mask.shape[-1], dtype=torch.long, device=sim.device)

        return loss


    def forward(self, embedding, abs_coords, patch_mask) -> Tensor:
        # embedding.shape = (b, p, c)
        # b = batchsize
        # p = number of patches
        # c = number of embedding

        coord_diff = abs_coords[:, :, None] - abs_coords[:, None]
        e0 = embedding[:, :, None]
        e1 = embedding[:, None, :]

        # compute all pairwise distaances of anchor embeddings
        diff = (e0 - e1)
        diff[..., :2] += coord_diff

        dist = diff.norm(2, dim=-1) / self.temperature

        sim = self.similarity(dist)

        return self.anchor_loss(sim, patch_mask)

class AnchorSigmoidPlusLoss(AnchorPlusLoss):
    def similarity(self, distance):
        return 1-(distance - self.anchor_radius).sigmoid()

class AnchorEuklPlusLoss(AnchorPlusLoss):
    def similarity(self, distance):
        return -distance

class AnchorRBFPlusLoss(AnchorPlusLoss):
    def similarity(self, distance):
        return (-distance.pow(2)).exp()

    def anchor_loss(self, sim, patch_mask):

        loss = 0
        num_patches = sim.shape[-1]

        # copy sim array and mask all values that are not negative pairs with -inf
        neg_masked_sim = sim.masked_fill(patch_mask != -1, float('-inf'))
        pos_masked_sim = sim.masked_fill(patch_mask !=  1, float('-inf'))

        # create mask of all columns that contain positive or negative pairs
        training_mask = (patch_mask<0).any(dim=2) * (patch_mask>0).any(dim=2)
        # fold column and batch dimension
        training_mask = training_mask.view(-1)

        # remove all columns that have no positive or negative pairs
        pos_masked_sim = pos_masked_sim.view(-1, num_patches)[training_mask]
        neg_masked_sim = neg_masked_sim.view(-1, num_patches)[training_mask]
        patch_mask = patch_mask.view(-1, num_patches)[training_mask]

        target = torch.zeros(patch_mask.shape[-1], dtype=torch.long, device=sim.device)

        print((patch_mask == 1).long().sum(dim=-1), (patch_mask == -1).long().sum(dim=-1), (patch_mask).long().sum(dim=-1))

        positive_sample = torch.logsumexp(pos_masked_sim, dim=1, keepdim=True)
        negative_samples = neg_masked_sim

        sample = torch.cat((positive_sample,
                            negative_samples), dim=-1)
        # print(sample)
        # loss = self.xnt_loss(sample, target[:len(sample)])
        loss = self.xnt_loss(sample[0:1], target[0:1])

        return loss