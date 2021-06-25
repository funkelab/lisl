from genericpath import exists
import os
import logging
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from torch import optim
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
import random

import pytorch_lightning as pl
from lisl.pl.model import MLP, get_unet_kernels, PatchedResnet
from funlib.learn.torch.models import UNet
from mipnet.models.unet import UNet2d
from deeplabv3plus.network.modeling import deeplabv3plus_resnet101
from stardist.matching import matching, matching_dataset

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


from radam import RAdam
import time
import numpy as np
from skimage.io import imsave
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

torch.autograd.set_detect_anomaly(True)



class SSLUnetTrainer(pl.LightningModule, BuildFromArgparse):
    def __init__(self,
                 in_channels,
                 out_channels,
                 initial_lr,
                 lr_milestones=(100)):

        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.initial_lr = initial_lr
        self.lr_milestones = list(int(_) for _ in lr_milestones)
        self.save_hyperparameters()
        self.build_models()
        self.build_loss()

    @staticmethod
    def add_model_specific_args(parser):
        parser.add_argument('--in_channels', default=512, type=int)
        parser.add_argument('--out_channels', default=3, type=int)
        parser.add_argument('--initial_lr', default=0.0001, type=float)
        parser.add_argument('--loss_direction_vector_file', type=str, default="misc/direction_vectors.npy")
        parser.add_argument('--lr_milestones', nargs='*', default=[10000, 20000])

        return parser

    def forward(self, x):
        return self.model(x)

    def build_models(self):
        # self.model = UNet(in_channels=self.in_channels,
        #                   num_fmaps=self.out_channels,
        #                   fmap_inc_factor=64,
        #                   downsample_factors=[[2, 2], [2, 2], [2, 2]],
        #                   kernel_size_down=[[[3, 3], [3, 3]]]*4,
        #                   kernel_size_up=[[[3, 3], [3, 3]]]*3,
        #                   padding='same')

        self.model = UNet2d(self.in_channels, self.out_channels, pad_convs=True)
        
        # self.model = deeplabv3plus_resnet101(num_classes=self.out_channels)
        # ptweights = self.model.backbone.conv1.weight.mean(dim=1, keepdim=True).data
        # self.model.backbone.conv1 = nn.Conv2d(self.in_channels, 64, kernel_size=7, padding=3, bias=False)
        # if self.in_channels == 1:
        #     self.model.backbone.conv1.weight.data = ptweights


    def build_loss(self, ):
        self.loss = nn.CrossEntropyLoss(weight=torch.Tensor([2., 1., 1.]))

    def training_step(self, batch, batch_nb):

        # ensure that model is in training mode
        self.model.train()
        raw, embedding, target, gt = batch
        y = self.forward(embedding)
        loss = self.loss(y, target)

        self.log('train/loss', loss.detach(), on_step=True,
                 on_epoch=True, prog_bar=True, logger=True)
        
        if self.global_step % 1000 == 0:
            os.makedirs("img", exist_ok=True)
            pred_grid = make_grid(y.detach().cpu().softmax(1), nrow=len(y)).permute(1, 2, 0).numpy()
            raw_grid = make_grid(raw.detach().cpu(), normalize=True, nrow=len(
                y), scale_each=True).permute(1, 2, 0).numpy()
            target_grid = make_grid(target.detach().cpu()[:, None].float(), normalize=True, nrow=len(
                y), scale_each=True).permute(1, 2, 0).numpy()
            gt_stack = torch.stack([torch.from_numpy(label2color(gt.detach().cpu().numpy()[i])) for i in range(len(gt))])
            gt_grid = make_grid(gt_stack, nrow=len(y))[
                :3].permute(1, 2, 0).numpy()
            grid = np.concatenate((pred_grid, raw_grid, target_grid, gt_grid), axis=0)
            imsave(f'img/{self.global_step:08}.png', grid)

        return {'loss': loss}

    def validation_step(self, batch, batch_nb):

        # set model to validation mode
        self.model.eval()

        raw, embedding, target, gt = batch
        y = self.forward(embedding)

        loss = self.loss(y, target)
        self.log('val/loss', loss.detach(), on_step=True,
                 on_epoch=True, prog_bar=True, logger=True)

        y_amax = y.argmax(1)[0]
        inner = (y_amax == 1).detach().cpu().numpy()
        background = (y_amax == 2).detach().cpu().numpy()
        y_seg = compute_3class_segmentation(inner, background)


        # set model back to training mode
        self.model.train()
        return {'val/loss': loss, "gt": gt[0].detach().cpu().numpy(), "yseg": y_seg}

    def validation_epoch_end(self, outs):

        taus = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        stats = [matching_dataset([a['gt'] for a in outs], 
                                  [a['yseg'] for a in outs],
                                  thresh=t, show_progress=False) for t in taus]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        for m in ('precision', 'recall', 'accuracy', 'f1', 'mean_true_score', 'mean_matched_score', 'panoptic_quality'):
            ax1.plot(taus, [s._asdict()[m] for s in stats], '.-', lw=2, label=m)
        ax1.set_xlabel(r'IoU threshold $\tau$')
        ax1.set_ylabel('Metric value')
        ax1.grid()
        ax1.legend()

        for m in ('fp', 'tp', 'fn'):
            ax2.plot(taus, [s._asdict()[m] for s in stats], '.-', lw=2, label=m)
        ax2.set_xlabel(r'IoU threshold $\tau$')
        ax2.set_ylabel('Number #')
        ax2.grid()
        ax2.legend();
        fig.savefig(f"iou{self.global_step:012}.png")
        plt.close('all')
        vislist = [torch.cat([torch.from_numpy(label2color(a['yseg'])),
                              torch.from_numpy(label2color(a['gt']))], dim=-2)[:3] for a in outs
                              if a['yseg'].shape[-2:] == (256, 256)]

        grid = make_grid(vislist).permute(1, 2, 0).numpy()
        imsave(f'val_seg_{self.global_step:012}.png', grid)

        for t, ms in zip(taus, stats):
            msdict = ms._asdict()
            for k in ('precision', 'recall', 'accuracy', 'f1', 'mean_true_score', 'mean_matched_score', 'panoptic_quality'):
                self.log(f'val/{k}_{t}', float(msdict[k]), prog_bar=((t in [0.9]) and (k in ['f1'])), logger=True)
        

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.initial_lr, weight_decay=0.0)
        scheduler = MultiStepLR(optimizer, milestones=self.lr_milestones, gamma=0.2)
        return [optimizer], [scheduler]

