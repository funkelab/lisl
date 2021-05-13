import os
import logging
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch import optim
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
import random

import pytorch_lightning as pl
from lisl.pl.model import MLP, get_unet_kernels, PatchedResnet
from funlib.learn.torch.models import UNet

from radam import RAdam
import time
import numpy as np
from skimage.io import imsave
from tiktorch.models.dunet import DUNet
import math

import itertools
import os
import copy
from lisl.datasets import Dataset
import zarr
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import stardist
from skimage import measure
from scipy import ndimage
from skimage.segmentation import watershed
from lisl.pl.visualizations import vis_anchor_embedding
from lisl.pl.utils import (adapted_rand, vis, 
    label2color, try_remove, BuildFromArgparse, offset_from_direction)
from ctcmetrics.seg import seg_metric
from sklearn.decomposition import PCA
from lisl.pl.evaluation import compute_3class_segmentation
from lisl.pl.loss import AnchorLoss
from lisl.pl.loss import AnchorLoss, SineAnchorLoss
from lisl.pl.loss_supervised import SupervisedInstanceEmbeddingLoss

from torch.optim.lr_scheduler import MultiStepLR
import h5py
from lisl.pl.utils import vis, offset_slice, label2color, visnorm
from vit_pytorch.spatialvit import SpatialViT

torch.autograd.set_detect_anomaly(True)

class SSLTrainer(pl.LightningModule, BuildFromArgparse):
    def __init__(self, loss_name="CPC",
                       unet_type="gp",
                       distance=8,
                       head_layers=4,
                       vit_depth=8,
                       encoder_layers=2,
                       ndim=2,
                       in_channels=1,
                       out_channels=18,
                       stride=2,
                       initial_lr=1e-4,
                       regularization=1e-4,
                       temperature=10,
                       temperature_decay=0.99,
                       pretrained_model=False,
                       resnet_size=18,
                       val_patch_inference_steps=None,
                       val_patch_inference_downsample=None,
                       loss_direction_vector_file="misc/direction_vectors.npy",
                       loss_distances_file="misc/distances.npy",
                       lr_milestones=(100)):

        super().__init__()

        self.save_hyperparameters()

        self.out_channels = out_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ndim = ndim
        self.last_val_log = -1
        self.head_layers = head_layers
        self.vit_depth = vit_depth
        self.encoder_layers = encoder_layers
        self.loss_name = loss_name
        self.unet_type = unet_type
        self.initial_lr = initial_lr
        self.lr_milestones = list(int(_) for _ in lr_milestones)
        self.regularization = regularization
        self.temperature = temperature
        self.temperature_decay = temperature_decay
        self.pretrained_model = pretrained_model
        self.resnet_size = resnet_size
        self.loss_direction_vector_file = loss_direction_vector_file
        self.loss_distances_file = loss_distances_file

        self.val_train_set_size = [10, 20, 50, 100, 200, 500, 1000]
        self.val_patch_inference_steps = val_patch_inference_steps
        self.val_patch_inference_downsample = val_patch_inference_downsample
        self.reset_val()

        self.distance = distance
        self.stride = stride
        self.embed_scale = 0.1
        self.build_models()
        self.build_loss()
        self.ndim = 2

    def reset_val(self):
        self.val_metrics = {}
        for n in self.val_train_set_size:
            self.val_metrics['validation_loss'] = []
            self.val_metrics[f'val_arand_{n}'] = []
            self.val_metrics[f'val_seg_{n}'] = []


    @staticmethod
    def add_model_specific_args(parser):
        # parser.add_argument('--n_workers', type=int, default=10)
        parser.add_argument('--head_layers', default=2, type=int)
        parser.add_argument('--encoder_layers', default=2, type=int)
        parser.add_argument('--vit_depth', type=int, default=0)
        parser.add_argument('--ndim', type=int, default=2)
        parser.add_argument('--out_channels', type=int, default=64)
        parser.add_argument('--distance', type=int, default=8)
        parser.add_argument('--in_channels', type=int, default=1)
        parser.add_argument('--initial_lr', type=float, default=1e-4)
        parser.add_argument('--regularization', type=float, default=1e-4)
        parser.add_argument('--loss_name', type=str, default="CPC")
        parser.add_argument('--unet_type', type=str, default="gp")
        parser.add_argument('--lr_milestones', nargs='*', default=[10000, 20000])
        parser.add_argument('--temperature', type=float, default=10)
        parser.add_argument('--temperature_decay', type=float, default=0.99)
        parser.add_argument('--pretrained_model', action='store_true')
        parser.add_argument('--val_patch_inference_steps', type=int, default=None)
        parser.add_argument('--val_patch_inference_downsample', type=int, default=None)
        parser.add_argument('--resnet_size', type=int, default=18)
        parser.add_argument('--loss_direction_vector_file', type=str, default="misc/direction_vectors.npy")
        parser.add_argument('--loss_distances_file', type=str, default="misc/distances.npy")

        return parser

    def forward(self, x):
        return self.model(x)[1]

    def forward_patches(self, x, abs_coords, img=None, vis=False):
        # expects input in the shape of
        # x.shape = 
        # (batch, patch_size, channels, patch_width, patch_height)
        b, p, c, pw, ph = x.shape
        z, out_0 = self.model.forward(x.view(-1, c, pw, ph))
        out_0 = out_0.view(b, p, self.out_channels)
        if abs_coords is None or self.spatial_transformer is None:
            return out_0, None
        z = z.view(b, p, -1)
        z, out_1 = self.spatial_transformer(z, out_0 + abs_coords)
        out_1 = out_1 - abs_coords
        out_1 = out_1.view(b, p, self.out_channels)
        return out_0, out_1

    def build_models(self):
        self.model = PatchedResnet(self.in_channels,
                                    self.out_channels,
                                    pretrained=self.pretrained_model,
                                    resnet_size=self.resnet_size)

        self.spatial_transformer = None
        if self.vit_depth > 0:
            self.spatial_transformer = SpatialViT(2,
                                                self.model.features_in_last_layer,
                                                self.vit_depth,
                                                64)

    def build_loss(self, ):
        self.validation_loss = SupervisedInstanceEmbeddingLoss(30.)
        # self.anchor_loss = SineAnchorLoss(self.temperature,
        #                                   self.loss_direction_vector_file,
        #                                   self.loss_distances_file)
        self.anchor_loss = AnchorLoss(self.temperature)

    def training_step(self, batch, batch_nb):

        # ensure that model is in training mode
        self.model.train()
        
        x, patches, abs_coords, patch_matches, mask = batch

        emb_0, emb_1 = self.forward_patches(patches,
                                            abs_coords,
                                            x,
                                            vis=(self.global_step % 100 == 0))

        if self.global_step % 1000 == 0 or (self.global_step < 2000 and self.global_step % 100 == 0):
            if emb_0.requires_grad:
                def log_hook(grad_input):
                    img_directory = os.path.abspath(os.path.join(self.logger.log_dir,
                                              os.pardir,
                                              os.pardir,
                                              "img"))
                    
                    os.makedirs(img_directory, exist_ok=True)
                    for b in range(len(emb_0)):
                        for c in range(0, emb_0.shape[-1]-1, 2):
                            vis_anchor_embedding(emb_0[b, ..., c:c+2].detach().cpu().numpy(),
                                abs_coords[b].detach().cpu().numpy(),
                                x[b].detach().cpu().numpy(),
                                grad=-grad_input[b, ..., c:c+2].detach().cpu().numpy(),
                                output_file=[f"{img_directory}/grad_{self.global_step}_{b}_{c}.jpg"])
                    handle.remove()

                handle = emb_0.register_hook(log_hook)

            if emb_1 is not None and emb_1.requires_grad:
                def log_hook(grad_input):
                    img_directory = os.path.abspath(os.path.join(self.logger.log_dir,
                                                                 os.pardir,
                                                                 os.pardir,
                                                                 "img"))

                    os.makedirs(img_directory, exist_ok=True)
                    for b in range(len(emb_1)):
                        for c in range(0, emb_1.shape[-1]-1, 2):
                            vis_anchor_embedding(emb_1[b, ..., c:c+2].detach().cpu().numpy(),
                                                 abs_coords[b].detach(
                            ).cpu().numpy(),
                                x[b].detach().cpu().numpy(),
                                grad=-grad_input[b, ..., c:c +
                                                 2].detach().cpu().numpy(),
                                output_file=[f"{img_directory}/vi_grad_{self.global_step}_{b}_{c}.jpg"])
                    handle.remove()

                handle = emb_1.register_hook(log_hook)

            
        # detach masked  boundary patches
        emb_0 = ((mask[..., None]).float() * emb_0.detach()) + \
            ((~mask[..., None]).float() * emb_0)
        anchor_loss = self.anchor_loss(emb_0, abs_coords, patch_matches)
        if emb_1 is not None:
            emb_1 = ((mask[..., None]).float() * emb_1.detach()) + ((~mask[..., None]).float() * emb_1)
            anchor_loss = anchor_loss + \
                self.anchor_loss(emb_1, abs_coords, patch_matches)

        self.log('anchor_loss', anchor_loss.detach(), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('anchor_loss_temperature', self.anchor_loss.temperature, on_step=True, prog_bar=True, logger=True)

        if self.regularization > 0.:
            reg_loss = self.regularization * emb_0.norm(2, dim=-1).sum()
            if emb_1 is not None:
                reg_loss = reg_loss + self.regularization * \
                    emb_1.norm(2, dim=-1).sum()
            loss = anchor_loss + reg_loss
            self.log('reg_loss', reg_loss.detach(), on_step=True, prog_bar=True, logger=True)
        else:
            loss = anchor_loss

        # loss, y, y_hat = self.loss_anchor(y, embedding)
        tensorboard_logs = {'train_loss': loss.detach()}
        tensorboard_logs['iteration'] = self.global_step
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb):

        # set model to validation mode
        self.model.eval()
        x, patches, abs_coords, patch_matches, mask, y = batch

        with torch.no_grad():

            if self.val_patch_inference_downsample is not None:
                patches = patches[:, ::self.val_patch_inference_downsample]
                abs_coords = abs_coords[:, ::self.val_patch_inference_downsample]
                patch_matches = patch_matches[:, ::self.val_patch_inference_downsample,
                                                 ::self.val_patch_inference_downsample]

            if self.val_patch_inference_steps is not None:
                embedding = self.sliced_cpu_forward_patches(
                    patches, abs_coords)
                abs_coords = abs_coords.cpu()
                patch_matches = patch_matches.cpu()
            else:
                embedding, _ = self.forward_patches(patches, abs_coords)

            loss = self.anchor_loss(embedding, abs_coords, patch_matches)
            self.log('val_anchor_loss', loss.detach(), on_epoch=True, prog_bar=False, logger=True)
            for margin in [1., 5., 10, 20., 40]:
                self.validation_loss.push_margin = margin
                absoute_embedding = self.anchor_loss.absoute_embedding(embedding, abs_coords)
                pull_loss, push_loss = self.validation_loss(absoute_embedding, abs_coords, y, split_pull_push=True)
                self.log(f'val_clustering_loss_margin_pull_{margin}', pull_loss.detach(), on_epoch=True, prog_bar=False, logger=True)
                self.log(f'val_clustering_loss_margin_push_{margin}', push_loss.detach(), on_epoch=True, prog_bar=False, logger=True)
                self.log(f'val_clustering_loss_margin_both_{margin}', (pull_loss+push_loss).detach(), on_epoch=True, prog_bar=False, logger=True)

        # set model back to training mode
        self.model.train()

        return {'val_loss': loss}

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.initial_lr, weight_decay=0.01)
        scheduler = MultiStepLR(optimizer, milestones=self.lr_milestones, gamma=0.2)
        return [optimizer], [scheduler]

    def sliced_cpu_forward_patches(self, patches, abs_coords):
        vpis = self.val_patch_inference_steps
        # apply forward patch on small slices to  
        # reduce memory footprint
        # accumulate results on cpu
        embedding = []
        for b in range(patches.shape[0]):
            e_slice = []
            for c in range(0, patches.shape[1], vpis):
                p, _ = self.forward_patches(patches[b:b+1, c:c+vpis], None)
                e_slice.append(p.cpu())
            e_slice = torch.cat(e_slice, dim=1)
            embedding.append(e_slice)
        embedding = torch.cat(embedding, dim=0)

        return embedding

    def log_now(self, val=False):

        if val:
            if self.last_val_log == self.global_step:
                return False
            else:
                self.last_val_log = self.global_step
                return True

        if self.global_step > 1024:
            return self.global_step % 2048 == 0
        else:
            return self.global_step % 64 == 0

