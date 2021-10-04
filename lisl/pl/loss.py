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
import numpy as np
from lisl.pl.utils import cluster_embeddings, label2color
from skimage.io import imsave
import time
    
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

    def distance_fn(self, embedding, abs_coords):
        e0 = embedding[:, :, None]
        e1 = embedding[:, None]
        coord_diff = abs_coords[:, :, None] - abs_coords[:, None]        
        diff = (e0 - e1) + coord_diff
        return diff.norm(2, dim=-1)

    def nonlinearity(self, distance):
        return 1 - (-distance.pow(2) / self.temperature).exp()

    def forward(self, embedding, abs_coords, patch_mask) -> Tensor:
        # compute all pairwise distances of anchor embeddings
        dist = self.distance_fn(embedding, abs_coords)
        # dist.shape = (b, p, p)

        nonlinear_dist = self.nonlinearity(dist)
        
        # only matched patches (e.g. patches close in proximity)
        # contripute to the loss
        nonlinear_dist = nonlinear_dist[patch_mask == 1]

        return nonlinear_dist.sum()

    def absoute_embedding(self, embedding, abs_coords):
        return embedding + abs_coords

    def multiply_temperature(self, gamma):
        if self.temperature > 1.:
            self.temperature *= gamma

class AnchorPlusContrastiveLoss(AnchorLoss):

    def __init__(self, *args) -> None:
        super().__init__(*args)
        self.ce = torch.nn.CrossEntropyLoss(ignore_index=-1)
        self.weight = 10.
    
    def forward(self, embedding, contr_emb, abs_coords, patch_mask) -> Tensor:
        # compute all pairwise distances of anchor embeddings
        dist = self.distance_fn(embedding, abs_coords)
        # dist.shape = (b, p, p)

        nonlinear_dist = self.nonlinearity(dist)
        # only matched patches (e.g. patches close in proximity)
        # contripute to the loss
        nonlinear_dist = nonlinear_dist[patch_mask == 1]

        loss = nonlinear_dist.mean()

        if contr_emb is None:
            return loss

        cluster_labels = cluster_embeddings(embedding + abs_coords)
        contr_emb = F.normalize(contr_emb, dim=-1)
        cum_mean_clusters = []
        
        for b in range(len(embedding)):
            with torch.no_grad():
                mean_clusters = [contr_emb[b, cluster_labels[b]==i].mean(axis=0) for i in np.unique(cluster_labels[b]) if i >= 0]
                mean_clusters = torch.stack(mean_clusters, dim=-1)
                cum_mean_clusters.append(mean_clusters)

        cum_mean_clusters = torch.cat(cum_mean_clusters, dim=-1)
        stacked_contr_emb = contr_emb.view(-1, cum_mean_clusters.shape[0])
        logits = torch.matmul(stacked_contr_emb, cum_mean_clusters)
        target = torch.from_numpy(np.concatenate(cluster_labels, axis=0)).long().to(logits.device)
        bce_loss = self.ce(logits, target)
        loss += self.weight * bce_loss

        return loss

class LinearAnchorLoss(AnchorLoss):
    def nonlinearity(self, distance):
        return distance

class LinearAnchorLoss(AnchorLoss):
    def nonlinearity(self, distance):
        return distance

class SigmoidAnchorLoss(AnchorLoss):
    def nonlinearity(self, distance):
        return (distance - self.temperature).sigmoid()

class SineAnchorLoss(AnchorLoss):

    def __init__(self, temperature, direction_vector_file, distances_file):
        super().__init__(temperature)
        dirs = np.loadtxt(direction_vector_file).astype(np.float32)
        direction_vectors = torch.from_numpy(dirs)
        dst = np.loadtxt(distances_file).astype(np.float32)
        dst = torch.from_numpy(dst)
        self.periodicity = ((2 * np.pi) / dst)[None, None]
        self.coord_transform = direction_vectors

    def distance_fn(self, embedding, abs_coords):
        # embedding.shape = (b, p, c)
        # abs_coords.shape = (b, p, 2)
        assert embedding.shape[-1] <= self.coord_transform.shape[0]
        z = self.absoute_embedding(embedding, abs_coords)
        return 1 - (-(z[:, :, None] - z[:, None]).pow(2) / self.temperature).exp()

    def absoute_embedding(self, embedding, abs_coords):
        ct = self.coord_transform[:embedding.shape[-1]].to(abs_coords.device)
        w = self.periodicity[..., :embedding.shape[-1]].to(abs_coords.device)
        transformed_coords = torch.einsum('cs,bps->bpc', ct, abs_coords)
        abs_embedding = embedding + transformed_coords
        # return flat toroidal coordinates
        return torch.cat((torch.sin(w * abs_embedding)/w, torch.cos(w * abs_embedding)/w), dim=-1)

    def nonlinearity(self, distance):
        return distance


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
