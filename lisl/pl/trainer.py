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
from skimage.io import imsave
from lisl.pl.visualizations import vis_anchor_embedding
from lisl.pl.utils import (adapted_rand, vis, 
    label2color, try_remove, BuildFromArgparse, offset_from_direction)
from ctcmetrics.seg import seg_metric
from sklearn.decomposition import PCA
from lisl.pl.evaluation import compute_3class_segmentation
from lisl.pl.loss import AnchorLoss
# from pytorch_lightning.losses.self_supervised_learning import CPCTask

from torch.optim.lr_scheduler import MultiStepLR
import h5py
from lisl.pl.utils import vis, offset_slice, label2color, visnorm

class SSLTrainer(pl.LightningModule, BuildFromArgparse):
    def __init__(self, loss_name="CPC",
                       unet_type="gp",
                       distance=8,
                       head_layers=4,
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
                       lr_milestones=(100)):

        super().__init__()

        self.save_hyperparameters()

        self.out_channels = out_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ndim = ndim
        self.last_val_log = -1
        self.head_layers = head_layers
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
        parser.add_argument('--ndim', type=int, default=2)
        parser.add_argument('--out_channels', type=int, default=64)
        parser.add_argument('--distance', type=int, default=8)
        parser.add_argument('--in_channels', type=int, default=1)
        parser.add_argument('--initial_lr', type=float, default=1e-4)
        parser.add_argument('--regularization', type=float, default=1e-4)
        parser.add_argument('--loss_name', type=str, default="CPC")
        parser.add_argument('--unet_type', type=str, default="gp")
        parser.add_argument('--lr_milestones', nargs='*', default=[10000, 50000])
        parser.add_argument('--temperature', type=float, default=10)
        parser.add_argument('--temperature_decay', type=float, default=0.99)
        parser.add_argument('--pretrained_model', action='store_true')
        parser.add_argument('--val_patch_inference_steps', type=int, default=None)
        parser.add_argument('--val_patch_inference_downsample', type=int, default=None)
        parser.add_argument('--resnet_size', type=int, default=18)
        return parser

    def forward(self, x):
        return self.model(x)

    def forward_patches(self, x, abs_coords=None, img=None, augmentation='nothing', vis=False):
        # expects input in the shape of
        # x.shape = 
        # (batch, patch_size, channels, patch_width, patch_height)
        b, p, c, pw, ph = x.shape

        if augmentation=='flip':
            # apply flips and transpose to imput patches
            inp = x.view(-1, c, pw, ph)
            inp_tp = inp.transpose(-2, -1).detach()
            inp_flip = torch.flip(inp, [-2, -1]).detach()
            inp_tp_flip = torch.flip(inp_tp, [-2, -1]).detach()

            # forward through model
            out = self.model(inp)
            out_tp = self.model(inp_tp)
            out_flip = self.model(inp_flip)
            out_tp_flip = self.model(inp_tp_flip)

            # flip and transpose output embeddings
            shuffle_index = [1, 0] + [_ for _ in range(2, out.shape[-1])]
            out_tp = out_tp[..., shuffle_index]
            out_flip[..., :2] *= -1.
            out_tp_flip[..., :2] *= -1.
            out_tp_flip = out_tp_flip[..., shuffle_index]

            out = (out + out_tp + out_flip + out_tp_flip) / 4.

        elif augmentation=='rotation':

            with torch.no_grad():
                inp_r0 = x.view(-1, c, pw, ph)
                inp_r1 = torch.rot90(inp_r0, 1, [-2, -1]).detach()
                inp_r2 = torch.rot90(inp_r0, 2, [-2, -1]).detach()
                inp_r3 = torch.rot90(inp_r0, 3, [-2, -1]).detach()

            # forward through model
            out_r0 = self.model(inp_r0)
            out_r1 = self.model(inp_r1)
            out_r2 = self.model(inp_r2)
            out_r3 = self.model(inp_r3)

            # rotate output embeddings
            shuffle_index = [1, 0] + [_ for _ in range(2, out_r0.shape[-1])]
            out_r1 = out_r1[..., shuffle_index]
            out_r1[..., 0] *= -1.
            out_r2[..., :2] *= -1.
            out_r3 = out_r3[..., shuffle_index]
            out_r3[..., 1] *= -1.

            out = (out_r0 + out_r1 + out_r2 + out_r3) / 4.
            
        elif augmentation=='rotflip':

            with torch.no_grad():
                inp_r0 = x.view(-1, c, pw, ph)
                inp_tp = inp_r0.transpose(-2, -1).detach()
                inp_flip = torch.flip(inp_r0, [-2, -1]).detach()
                inp_tp_flip = torch.flip(inp_tp, [-2, -1]).detach()
                inp_r1 = torch.rot90(inp_r0, 1, [-2, -1]).detach()
                inp_r2 = torch.rot90(inp_r0, 2, [-2, -1]).detach()
                inp_r3 = torch.rot90(inp_r0, 3, [-2, -1]).detach()

            # forward through model
            out_r0 = self.model(inp_r0)
            out_r1 = self.model(inp_r1)
            out_r2 = self.model(inp_r2)
            out_r3 = self.model(inp_r3)
            out_tp = self.model(inp_tp)
            out_flip = self.model(inp_flip)
            out_tp_flip = self.model(inp_tp_flip)


            # rotate output embeddings
            shuffle_index = [1, 0] + [_ for _ in range(2, out_r0.shape[-1])]
            out_r1 = out_r1[..., shuffle_index]
            out_r1[..., 0] *= -1.
            out_r2[..., :2] *= -1.
            out_r3 = out_r3[..., shuffle_index]
            out_r3[..., 1] *= -1.

            out_tp = out_tp[..., shuffle_index]
            out_flip[..., :2] *= -1.
            out_tp_flip[..., :2] *= -1.
            out_tp_flip = out_tp_flip[..., shuffle_index]

            # average embeddings
            out = (2 * out_r0 + out_r1 + out_r2 + out_r3 + out_tp + out_flip + out_tp_flip) / 8. 
        else:
            out = self.model(x.view(-1, c, pw, ph))

        out = out.view(b, p, self.out_channels)
        return out

    def build_models(self):
        self.model = PatchedResnet(self.in_channels,
                                    self.out_channels,
                                    pretrained=self.pretrained_model,
                                    resnet_size=self.resnet_size)

    def build_loss(self, ):
        self.anchor_loss = AnchorLoss(self.temperature)

    def training_step(self, batch, batch_nb):
        x, patches, abs_coords, patch_matches, mask = batch

        embedding = self.forward_patches(patches, abs_coords, x, vis=(self.global_step % 100 == 0))

        if self.global_step % 1000 == 0 or (self.global_step < 2000 and self.global_step % 100 == 0):

            # save model
            model_directory = os.path.abspath(os.path.join(self.logger.log_dir,
                                                          os.pardir,
                                                          os.pardir,
                                                          "model"))
            model_save_path = os.path.join(model_directory, f"model_{self.global_step}.torch")
            os.makedirs(model_directory, exist_ok=True)
            torch.save(self.model.state_dict(), model_save_path)

            with torch.no_grad():
                img_directory = os.path.abspath(os.path.join(self.logger.log_dir,
                                                os.pardir,
                                                os.pardir,
                                                "img"))
                os.makedirs(img_directory, exist_ok=True)
                for b in range(len(embedding)):
                    for c in range(0, embedding.shape[-1]-1, 2):
                        vis_anchor_embedding(embedding[b, ..., c:c+2].detach().cpu().numpy(),
                            abs_coords[b].detach().cpu().numpy(),
                            x[b].detach().cpu().numpy(),
                            output_file=[f"{img_directory}/vis_{self.global_step}_{self.local_rank}_{b}.jpg",
                                         f"{img_directory}/vis_{self.global_step}_{self.local_rank}_{b}.pdf"])

            if embedding.requires_grad:
                def log_hook(grad_input):
                    img_directory = os.path.abspath(os.path.join(self.logger.log_dir,
                                              os.pardir,
                                              os.pardir,
                                              "img"))
                    os.makedirs(img_directory, exist_ok=True)
                    for b in range(len(embedding)):
                        for c in range(0, embedding.shape[-1]-1, 2):
                            vis_anchor_embedding(embedding[b, ..., c:c+2].detach().cpu().numpy(),
                                abs_coords[b].detach().cpu().numpy(),
                                x[b].detach().cpu().numpy(),
                                grad=-grad_input[b, ..., c:c+2].detach().cpu().numpy(),
                                output_file=[f"{img_directory}/grad_{self.global_step}_{b}_{c}.jpg",
                                             f"{img_directory}/grad_{self.global_step}_{b}_{c}.pdf"])
                    handle.remove()

                handle = embedding.register_hook(log_hook)


            # max_x = int(abs_coords[..., 0].max().cpu().numpy())
            # max_y = int(abs_coords[..., 1].max().cpu().numpy())

            # origin = abs_coords.shape[1] // 2
            # cx, cy = abs_coords[0, origin]

            # for center_idx in [origin, origin+10, origin+20]:
            #     ds = 10
            #     heatmap = np.empty((100//ds, 100//ds))
            #     tmpemb = embedding[:1].detach()
            #     with torch.no_grad():
            #         stride = 1
            #         for i, x in enumerate(np.arange(-5, 5, 0.1*ds)):
            #             for j, y in enumerate(np.arange(-5, 5, 0.1*ds)):
            #                 tmpemb[0, center_idx, 0] = x - cx
            #                 tmpemb[0, center_idx, 1] = y - cy
            #                 heatmap[i, j] = self.anchor_loss(tmpemb, abs_coords[:1], patch_matches[:1]).detach().cpu().numpy()
            #                 if i ==0 and j==0:
            #                     heatmap[:] = heatmap[i, j]
            #                 print(tmpemb[0, center_idx], heatmap[i, j], x, y, i, j)

            #             imsave(f"{img_directory}/heatmap_{self.global_step}_{center_idx}.png", heatmap)



        # detach masked  boundary patches
        embedding = ((mask[..., None]).float() * embedding.detach()) + ((~mask[..., None]).float() * embedding)
        anchor_loss = self.anchor_loss(embedding, abs_coords, patch_matches)

        # if self.global_step < 1000:
        #     self.anchor_loss.multiply_temperature(1.003)
        # else:
        #     self.anchor_loss.multiply_temperature(self.temperature_decay)

        self.log('anchor_loss', anchor_loss.detach(), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('anchor_loss_temperature', self.anchor_loss.temperature, on_step=True, prog_bar=True, logger=True)

        if self.regularization > 0.:
            reg_loss = self.regularization * embedding.norm(2, dim=-1).sum()
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

        import os, psutil
        process = psutil.Process(os.getpid())
        from inspect import currentframe, getframeinfo

        with torch.no_grad():
            print(getframeinfo(currentframe()).filename, getframeinfo(currentframe()).lineno, process.memory_info().rss)
            if self.val_patch_inference_downsample is not None:
                patches = patches[:, ::self.val_patch_inference_downsample]
                abs_coords = abs_coords[:, ::self.val_patch_inference_downsample]
                patch_matches = patch_matches[:, ::self.val_patch_inference_downsample,
                                                 ::self.val_patch_inference_downsample]
            print(getframeinfo(currentframe()).filename, getframeinfo(currentframe()).lineno, process.memory_info().rss)
            if self.val_patch_inference_steps is not None:
                print(getframeinfo(currentframe()).filename, getframeinfo(currentframe()).lineno, process.memory_info().rss)
                print(getframeinfo(currentframe()).filename, getframeinfo(currentframe()).lineno, process.memory_info().rss)
                embedding = self.sliced_cpu_forward_patches(patches)
                print(getframeinfo(currentframe()).filename, getframeinfo(currentframe()).lineno, process.memory_info().rss)
                abs_coords = abs_coords.cpu()
                patch_matches = patch_matches.cpu()
                print(getframeinfo(currentframe()).filename, getframeinfo(currentframe()).lineno, process.memory_info().rss)
            else:
                embedding = self.forward_patches(patches)

            print(getframeinfo(currentframe()).filename, getframeinfo(currentframe()).lineno, process.memory_info().rss)

            loss = self.anchor_loss(embedding, abs_coords, patch_matches)

            print(getframeinfo(currentframe()).filename, getframeinfo(currentframe()).lineno, process.memory_info().rss)
            
            self.log('val_loss', loss.detach(), on_step=True, on_epoch=True, prog_bar=False, logger=True)

        print(getframeinfo(currentframe()).filename, getframeinfo(currentframe()).lineno, process.memory_info().rss)

        # set model back to training mode
        self.model.train()

        return {'val_loss': loss}

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.initial_lr, weight_decay=0.01)
        scheduler = MultiStepLR(optimizer, milestones=self.lr_milestones, gamma=0.1)
        return [optimizer], [scheduler]

    def sliced_cpu_forward_patches(self, patches):
        vpis = self.val_patch_inference_steps
        # apply forward patch on small slices to  
        # reduce memory footprint
        # accumulate results on cpu
        embedding = []
        for b in range(patches.shape[0]):
            e_slice = []
            for c in range(0, patches.shape[1], vpis):
                e_slice.append(self.forward_patches(patches[b:b+1, c:c+vpis]).cpu())
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

