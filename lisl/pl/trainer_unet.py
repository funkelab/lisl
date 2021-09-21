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
from ignite.handlers.param_scheduler import PiecewiseLinear
import random
import operator

import pytorch_lightning as pl
from lisl.pl.model import MLP, get_unet_kernels, PatchedResnet, FNC, Deeplab
from funlib.learn.torch.models import UNet
from mipnet.models.unet import UNet2d
from deeplabv3plus.network.modeling import deeplabv3plus_resnet101
from stardist.matching import matching, matching_dataset
from stardist.geometry import polygons_to_label
from stardist import non_maximum_suppression

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import json

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
                 segmentation_type,
                 architecture="unet",
                 model_checkpoint=None,
                 fix_backbone_until=0,
                 lr_milestones=(100),
                 finetuning=False,
                 finetuning_unfreezing_interval=100,
                 finetuning_niterations=1000):

        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.initial_lr = initial_lr
        self.architecture = architecture
        self.model_checkpoint = model_checkpoint
        self.fix_backbone_until = fix_backbone_until
        self.lr_milestones = list(int(_) for _ in lr_milestones)
        self.finetuning = finetuning
        self.finetuning_unfreezing_interval = finetuning_unfreezing_interval
        self.finetuning_niterations = finetuning_niterations
        self.alpha = 0.01
        self.save_hyperparameters()
        self.build_models()
        self.segmentation_type = segmentation_type
        self.build_loss()

    @staticmethod
    def add_model_specific_args(parser):
        parser.add_argument('--in_channels', default=512, type=int)
        parser.add_argument('--out_channels', default=3, type=int)
        parser.add_argument('--initial_lr', default=0.0001, type=float)
        parser.add_argument('--segmentation_type', default="threeclass")
        parser.add_argument('--architecture', default="unet")
        parser.add_argument('--finetuning', action='store_true', help="uses Freez + STLR + disc (see https://arxiv.org/abs/1801.06146)")
        parser.add_argument('--finetuning_unfreezing_interval', type=int)
        parser.add_argument('--finetuning_niterations', type=int)
        parser.add_argument('--model_checkpoint', default=None)
        parser.add_argument('--fix_backbone_until', default=0, type=int)
        parser.add_argument('--loss_direction_vector_file', type=str, default="misc/direction_vectors.npy")
        parser.add_argument('--lr_milestones', nargs='*', default=[10, 15])

        return parser

    def forward(self, x):
        return self.model(x)

    def build_models(self):
        if self.architecture == "unet":
            self.model = UNet2d(self.in_channels, self.out_channels,
                                pad_convs=True, depth=3)
        elif self.architecture == "fcn":
            self.model = FNC(self.in_channels, self.out_channels,
                             checkpoint=self.model_checkpoint)

            if self.finetuning:
                self.layer_parameters = self.model.get_parameters_per_layer()

            if self.fix_backbone_until >0:
                for param in self.model.model.backbone.parameters():
                    param.requires_grad = False

        elif self.architecture == "deeplab":
            self.model = Deeplab(self.in_channels, self.out_channels,
                                 checkpoint=self.model_checkpoint)

            if self.finetuning:
                self.layer_parameters = self.model.get_parameters_per_layer()

            if self.fix_backbone_until >0:
                for param in self.model.model.backbone.parameters():
                    param.requires_grad = False

    def build_loss(self):
        if self.segmentation_type == "threeclass":
            self.loss = nn.CrossEntropyLoss(weight=torch.Tensor([2., 1., 1.]))
        elif self.segmentation_type == "stardist":
            self.loss = nn.L1Loss()

    def training_step(self, batch, batch_nb):

        # ensure that model is in training mode
        self.model.train()
        raw, embedding, target, gt = batch
        y = self.forward(embedding)
        loss = self.loss(y, target)

        self.log('train/loss', loss.detach(), on_step=True,
                 on_epoch=True, prog_bar=True, logger=True)

        if self.global_step % 1000 == 0 or (self.global_step < 1000 and self.global_step % 100 == 0):
            os.makedirs("img", exist_ok=True)
            if self.segmentation_type == "threeclass":
                pred_grid = make_grid(y.detach().cpu()[:, :3].softmax(1), nrow=len(y)).permute(1, 2, 0).numpy()
            elif self.segmentation_type == "stardist":
                pred_grid = make_grid(y.detach().cpu()[:, [0, 8, -1]], nrow=len(y), normalize=True).permute(1, 2, 0).numpy()
            
            raw_grid = make_grid(raw.detach().cpu(), normalize=True, nrow=len(
                y), scale_each=True).permute(1, 2, 0).numpy()
            if self.segmentation_type == "threeclass":
                target_grid = make_grid(target.detach().cpu()[:, None].float(), normalize=True, nrow=len(
                    y), scale_each=True).permute(1, 2, 0).numpy()
            elif self.segmentation_type == "stardist":
                target_grid = make_grid(target.detach().cpu()[:, [0, 8, -1]].float(), normalize=True, nrow=len(
                    y), scale_each=True).permute(1, 2, 0).numpy()
            gt_stack = torch.stack([torch.from_numpy(label2color(gt.detach().cpu().numpy()[i])) for i in range(len(gt))])
            gt_grid = make_grid(gt_stack, nrow=len(y))[
                :3].permute(1, 2, 0).numpy()
            grid = np.concatenate((pred_grid, raw_grid, target_grid, gt_grid), axis=0)
            imsave(f'img/{self.global_step:08}.png', grid)

        if self.fix_backbone_until > 0 and \
                self.fix_backbone_until == self.global_step:
                    print(f"unfreezing backbone in iteration {self.global_step}")
                    for param in self.model.model.backbone.parameters():
                        param.requires_grad = True

        if self.finetuning:
            unfreezing_index = (self.global_step // self.finetuning_unfreezing_interval) - 1
            # unpack all names that need to be unfrozen
            unfreezing_layers = [_ for sublist in self.layer_parameters[unfreezing_index:] for _ in sublist]

            # Freeze: unfreeze layers sequentially
            for name, param in self.model.model.named_parameters():
                param.require_grad = name in unfreezing_layers

            # STLR: adjust learning rate of all layers (slanted triangular learning rate)
            self.lr_schedulers().step()

            # disc: set unique learning rate for every layer
            for param_group in self.optimizers().param_groups:
                layer_index = int(param_group["name"])
                param_group["lr"] = param_group["lr"] / (1.2 ** layer_index)

        return {'loss': loss}

    def test_step(self, batch, batch_nb):

        raw, embedding, target, gt = batch
        y = self.forward(embedding)
        os.makedirs("test", exist_ok=True)

        if self.segmentation_type == "threeclass":
            y_amax = y.argmax(1)[0]
            inner = (y_amax == 1).detach().cpu().numpy()
            background = (y_amax == 2).detach().cpu().numpy()
            y_seg = compute_3class_segmentation(inner, background)
        elif self.segmentation_type == "stardist":
            # current implementation is not parallelized over batches
            assert y.shape[0] == 1
            # y.shape = (batch, nrays+1, width, height)
            width, height = y.shape[-2:]
            dist = y[0, :-1].detach().cpu().permute(1,2,0).numpy()
            prob = y[0, -1].detach().cpu().numpy()
            points, probi, disti = non_maximum_suppression(dist, prob)
            y_seg = polygons_to_label(disti, points, prob=probi, shape=(width, height))
        else:
            return NotImplementedError()

        color_gt = torch.from_numpy(label2color(gt[0].cpu().numpy())[:3])
        color_seg = torch.from_numpy(label2color(y_seg)[:3])
        
        vislist = [color_gt, color_seg] + [_.cpu()[None].expand(3, y.shape[-2], y.shape[-1]) for _ in y[0]]
        grid = make_grid(vislist, normalize=True, scale_each=True).permute(1, 2, 0).numpy()
        imsave(f'test/pred_{batch_nb:06}.png', grid)

        return {"gt": gt[0].detach().cpu().numpy(),
                "yseg": y_seg}

    def test_epoch_end(self, outs):

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

        try:
            vislist = [torch.cat([torch.from_numpy(label2color(a['yseg'])),
                                torch.from_numpy(label2color(a['gt']))], dim=-2)[:3] for a in outs]
                                # if a['yseg'].shape[-2:] == (690, 628)]

            grid = make_grid(vislist).permute(1, 2, 0).numpy()
            imsave(f'val_seg_{self.global_step:012}.png', grid)
        except:
            pass

        stats_dict = dict((t, ms._asdict()) for t, ms in zip(taus, stats))
        stats_dict["SEG score"] = np.mean([seg_metric(a['yseg'], a['gt']) for a in outs])

        with open(f"val_stats_{self.global_step:012}.json", "w") as stats_out:
            json.dump(stats_dict, stats_out)

    def configure_optimizers(self):
        if self.finetuning:
            parameter_groups = []
            for i, layer_p_names in reversed(list(enumerate(self.layer_parameters))):
                print("cfg opt", i, layer_p_names)
                group = {'name': str(i),
                         'params': [p for n, p in self.model.model.named_parameters() if n in layer_p_names]}
                parameter_groups.append(group)

            # Sanity check that we still have the same number of parameters
            assert sum(p.numel() for g in parameter_groups for p in g['params'])\
                == sum(p.numel() for p in self.model.parameters())

            optimizer = torch.optim.Adam(parameter_groups, lr=self.initial_lr)
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                            max_lr=self.initial_lr,
                                                            steps_per_epoch=1,
                                                            pct_start=0.1,
                                                            epochs=self.finetuning_niterations,
                                                            anneal_strategy='linear',
                                                            verbose=False)

        else:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.initial_lr, weight_decay=0.0)
            scheduler = MultiStepLR(optimizer, milestones=self.lr_milestones, gamma=0.2)
        return [optimizer], [scheduler]

