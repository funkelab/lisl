import os
import logging
from argparse import ArgumentParser
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
from lisl.pl.model import MLP, get_unet_kernels
from funlib.learn.torch.models import UNet

from radam import RAdam
from lisl.pl.dataset import MosaicDataModule
import time
import numpy as np

import functools
import inspect

# from pytorch_lightning.losses.self_supervised_learning import CPCTask

from torch.optim.lr_scheduler import MultiStepLR
from lisl.pl.utils import vis

from segmentation_models_pytorch.deeplabv3.model import DeepLabV3Plus


class SSLTrainer(pl.LightningModule):
    def __init__(self, loss_name="CPC", head_layers=4, encoder_layers=2, ndim=2, hidden_channels=64, in_channels=1, out_channels=3, initial_lr=1e-4):
        super().__init__()

        self.save_hyperparameters()

        self.out_channels = out_channels
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.ndim = ndim
        self.last_val_log = -1
        self.head_layers = head_layers
        self.encoder_layers = encoder_layers
        self.loss_name = loss_name
        self.initial_lr = initial_lr

        self.distance = 2
        self.stride = 4
        self.embed_scale = 0.1

        self.build_models()

    @staticmethod
    def add_model_specific_args(parser):
        # parser.add_argument('--n_workers', type=int, default=10)
        parser.add_argument('--head_layers', default=2, type=int)
        parser.add_argument('--encoder_layers', default=2, type=int)
        parser.add_argument('--ndim', type=int, default=2)
        parser.add_argument('--hidden_channels', type=int, default=64)
        parser.add_argument('--in_channels', type=int, default=1)
        parser.add_argument('--out_channels', type=int, default=3)
        parser.add_argument('--initial_lr', type=float, default=1e-4)
        parser.add_argument('--loss_name', type=str, default="CPC")

        return parser

    @classmethod
    def from_argparse_args(cls, args, **kwargs):

        if isinstance(args, ArgumentParser):
            args = cls.parse_argparser(args)
        params = vars(args)

        # we only want to pass in valid DataModule args, the rest may be user specific
        valid_kwargs = inspect.signature(cls.__init__).parameters
        datamodule_kwargs = dict(
            (name, params[name]) for name in valid_kwargs if name in params
        )
        datamodule_kwargs.update(**kwargs)

        return cls(**datamodule_kwargs)

    def forward(self, x):
        return self.unet(x)

    def build_models(self):

        dsf, ks = get_unet_kernels(self.ndim)

        self.unet = UNet(
            in_channels=self.in_channels,
            num_fmaps_out=self.hidden_channels,
            num_fmaps=32,
            fmap_inc_factor=2,
            downsample_factors=dsf,
            kernel_size_down=ks,
            kernel_size_up=ks,
            activation="GELU",
            num_heads=1,
            constant_upsample=True,
        )

        self.head = MLP(self.hidden_channels, self.out_channels, n_hidden_layers=self.head_layers,
                        n_hidden=self.hidden_channels, ndim=self.ndim)

        self.prediction_cnn = MLP(self.hidden_channels, self.hidden_channels,
                                  n_hidden_layers=self.encoder_layers,
                                  n_hidden=self.hidden_channels, ndim=2)
        self.target_cnn = MLP(self.hidden_channels, self.hidden_channels,
                              n_hidden_layers=self.encoder_layers,
                              n_hidden=self.hidden_channels, ndim=2)

    def loss_fn(self):
        return getattr(self, "loss_" + self.loss_name)

    def training_step(self, batch, batch_nb):
        x, y = batch

        embedding = self.forward(x)

        loss, y, y_hat = self.loss_fn()(y, embedding)
        tensorboard_logs = {'train_loss': loss.detach()}
        tensorboard_logs['iteration'] = self.global_step

        if self.log_now():
            self.logger.experiment.add_histogram(
                "inpainted_hist", y_hat[..., ::3, ::11, ::11].detach().cpu(), self.global_step)
            self.logger.experiment.add_histogram(
                "target_hist", y[..., ::3, ::11, ::11].detach().cpu(), self.global_step)
            self.logger.experiment.add_histogram(
                "embedding_hist", embedding[..., ::3, ::11, ::11].detach().cpu(), self.global_step)
            x_batch = torch.cat(tuple(torch.cat(tuple(vis(self.squeeze(x_0[c])) for c in range(
                x_0.shape[0])), dim=1) for x_0 in x.detach().cpu()), dim=-1)
            self.logger.experiment.add_image(f'x', x_batch, self.global_step)
            y_batch = torch.cat(tuple(torch.cat(tuple(vis(self.squeeze(y_0[c])) for c in range(
                y_0.shape[0])), dim=1) for y_0 in y.detach().cpu()), dim=-1)
            self.logger.experiment.add_image(f'y', y_batch, self.global_step)

            self.log_batch_images(x, y, y_hat, embedding, "train")

        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        x, y = batch
        embedding = self.forward(x)
        loss, y, y_hat = self.loss_fn()(y, embedding)

        if self.log_now(val=True):
            self.log_batch_images(x, y, y_hat, embedding, "val")

        return {'val_loss': loss}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.initial_lr)
        scheduler = MultiStepLR(optimizer, milestones=[4, 16, 64], gamma=0.5)
        return [optimizer], [scheduler]

    def squeeze(self, inp):
        if self.ndim == 2:
            return inp
        elif self.ndim == 3:
            return inp.max(dim=0)
        else:
            raise NotImplementedError()

    @staticmethod
    def bbox_to_slice(bbox, for_recurrent=False):
        sbox = []
        if for_recurrent:
            for i in [-1] + [j for j in range(len(bbox) - 1)]:
                b = bbox[i]
                sbox.append(slice(b["pos"], b["pos"] + b["width"]))
            return [slice(None), ] + sbox + [slice(None), ]
        else:
            for b in bbox:
                sbox.append(slice(b["pos"], b["pos"] + b["width"]))
            return [slice(None), slice(None)] + sbox

    def log_batch_images(self, x, y, y_hat, embedding, prefix):

        with torch.no_grad():

            x_0 = x[0].detach().cpu()
            c = 0
            x_batch = torch.cat(tuple(torch.cat(tuple(vis(self.squeeze(x_0[c])) for c in range(
                x_0.shape[0])), dim=1) for x_0 in x.detach().cpu()), dim=-1)
            self.logger.experiment.add_image(f'{prefix}_batch_not_croped', x_batch, self.global_step)

            embedding_batch = torch.cat(tuple(torch.cat(tuple(vis(self.squeeze(e_0[c])) for c in range(
                e_0.shape[0])), dim=1) for e_0 in embedding.detach().cpu()), dim=-1)
            self.logger.experiment.add_image(f'{prefix}_embedding', embedding_batch, self.global_step)

            y_hat_batch = torch.cat(tuple(torch.cat(tuple(vis(self.squeeze(y_hat_0[c])) for c in range(
                y_hat_0.shape[0])), dim=1) for y_hat_0 in y_hat.detach().cpu()), dim=-1)
            self.logger.experiment.add_image(f'{prefix}_y_hat', y_hat_batch, self.global_step)

            y_batch = torch.cat(tuple(torch.cat(tuple(vis(self.squeeze(y_0[c])) for c in range(
                y_0.shape[0])), dim=1) for y_0 in y.detach().cpu()), dim=-1)
            self.logger.experiment.add_image(f'{prefix}_y', y_batch, self.global_step)

            slices = []
            for c in range(len(x.shape)):
                if y_hat.shape[c] != x.shape[c] or c < 2:
                    reduction = x.shape[c] - y_hat.shape[c]
                    red_l = reduction // 2
                    red_r = reduction - red_l
                    slices.append(slice(red_l, x.shape[c] - red_r))
                else:
                    slices.append(slice(None))

            x = x[slices]

            if y.shape[-1] == x.shape[-1] and x.shape[-1] == y_hat.shape[-1]:
                cat_batch = torch.cat(tuple(
                    torch.cat(tuple(vis(self.squeeze(x_0[c])) for c in range(x_0.shape[0])) +
                              tuple(vis(self.squeeze(y_0[c])) for c in range(y_0.shape[0]))
                              + tuple(vis(self.squeeze(y_hat_0[c])) for c in range(y_hat_0.shape[0])), dim=1) for x_0, y_0, y_hat_0 in zip(x.detach().cpu(), y.detach().cpu(), y_hat.detach().cpu())), dim=2)
                self.logger.experiment.add_image(f'{prefix}_batch', cat_batch, self.global_step)

                x_0 = x[0].detach().cpu()
                y_0 = y[0].detach().cpu()
                y_hat_0 = y_hat[0].detach().cpu()

                inp_pred_diffs = []

                if self.ndim == 3:
                    for z in range(y_0.shape[1]):
                        pred = torch.stack(
                            tuple((torch.clamp(p, -1, 1) + 1) / 2 for p in y_hat_0[:, z]), dim=0)
                        targ = torch.stack(
                            tuple((torch.clamp(t, -1, 1) + 1) / 2 for t in y_0[:, z]), dim=0)
                        diff = torch.stack(tuple(torch.clamp((t - p) + 0.5, 0, 1)
                                                 for t, p in zip(y_0[:, z], y_hat_0[:, z])), dim=0)
                        mixed = torch.cat((vis(targ, normalize=False),
                                           vis(pred, normalize=False),
                                           vis(diff, normalize=False)), dim=-1)
                        inp_pred_diffs.append(mixed)
                else:
                    pred = torch.stack(
                        tuple((torch.clamp(p, -1, 1) + 1) / 2 for p in y_hat_0[:]), dim=0)
                    targ = torch.stack(
                        tuple((torch.clamp(t, -1, 1) + 1) / 2 for t in y_0[:]), dim=0)
                    if pred.shape[0] == 1:
                        targ = targ[:1]

                    diff = torch.stack(tuple(torch.clamp((t - p) + 0.5, 0, 1)
                                             for t, p in zip(y_0[:], y_hat_0[:])), dim=0)

                    mixed = torch.cat((vis(targ, normalize=False),
                                       vis(pred, normalize=False),
                                       vis(diff, normalize=False)), dim=-1)
                    inp_pred_diffs.append(mixed)

                self.logger.experiment.add_image(f'{prefix}_targ_pred_diff',
                                                 torch.cat(inp_pred_diffs, dim=-2), self.global_step)

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

    def loss_CPC(self, y, embedding):

        xrshift = random.randint(0, self.stride)
        yrshift = random.randint(0, self.stride)

        strided_embedding = embedding[..., xrshift::self.stride, yrshift::self.stride]

        preds = self.prediction_cnn(strided_embedding)
        targets = self.target_cnn(strided_embedding)
        targets = strided_embedding

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

        # generate the labels
        col_dim_i = h - self.distance - 1
        n = b * col_dim_i * w
        b1 = torch.arange(n) // (col_dim_i * w)
        c1 = torch.arange(n) % (col_dim_i * w)
        labels = b1 * h * w + (self.distance + 1) * w + c1
        labels = labels.to(logits.device)
        labels = labels.long()
        loss = nn.functional.cross_entropy(logits, labels)

        return loss, targets[:, :3], preds[:, :3]

    def loss_stardist(self, y, embedding, flipped_embedding):

        predicted_star_distances = torch.nn.functional.softplus(self.star_head(embedding))

        min_width = 10
        shift = np.random.randint(1, 16)
        bound = 0  # 100-shift

        scale = predicted_star_distances.shape[-1]

        psd_right = (predicted_star_distances[..., :-shift] * scale)
        psd_left = (predicted_star_distances[..., shift:] * scale)

        inner_right = torch.min(psd_left[:, 0] + shift, psd_right[:, 0])
        outer_right = torch.max(psd_left[:, 0] + shift, psd_right[:, 0])

        inner_left = torch.max(shift - psd_left[:, 1], - psd_right[:, 1])
        outer_left = torch.min(shift - psd_left[:, 1], - psd_right[:, 1])

        intersection = (inner_right - inner_left).abs()
        union = (outer_right - outer_left).abs()

        iou = intersection / (union + 1e-6)
        # consitency loss
        consitency = (4 * (-(iou - 0.5)**2 + 0.25))
        loss = consitency.mean()

        return loss, predicted_star_distances, iou[:, None]



