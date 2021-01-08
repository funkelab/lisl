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
from lisl.pl.model import MLP, get_unet_kernels, PatchedResnet50
from funlib.learn.torch.models import UNet

from radam import RAdam
import time
import numpy as np
from skimage.io import imsave

import functools
import inspect
from tiktorch.models.dunet import DUNet

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
from lisl.pl.utils import adapted_rand, vis, label2color, try_remove
from ctcmetrics.seg import seg_metric
from sklearn.decomposition import PCA
from lisl.pl.evaluation import compute_3class_segmentation
from lisl.pl.utils import adapted_rand, vis, label2color, try_remove
from lisl.pl.utils import offset_from_direction
# from pytorch_lightning.losses.self_supervised_learning import CPCTask

from torch.optim.lr_scheduler import MultiStepLR
from lisl.pl.utils import vis, offset_slice, label2color

# from segmentation_models_pytorch.deeplabv3.model import DeepLabV3Plus

class Deeplabv3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.unet = torch.hub.load('pytorch/vision:v0.6.0', 'deeplabv3_resnet101', pretrained=True)
        self.unet.classifier[4] = nn.Conv2d(256, out_channels, kernel_size=1, stride=1)
        self.unet.backbone.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, padding=3, bias=False)
        self.unet.aux_classifier[4] = nn.Conv2d(256, out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        return self.unet(x)['out']

class SSLTrainer(pl.LightningModule):
    def __init__(self, loss_name="CPC",
                       unet_type="gp",
                       distance=8,
                       head_layers=4,
                       encoder_layers=2,
                       ndim=2,
                       hidden_channels=64,
                       in_channels=1,
                       out_channels=3,
                       stride=2,
                       patchsize=16,
                       patchoverlap=5,
                       patchdilation=1,
                       initial_lr=1e-4):
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
        self.unet_type = unet_type
        self.initial_lr = initial_lr

        self.patchsize = patchsize
        self.patchoverlap = patchoverlap
        self.patchdilation = patchdilation

        self.val_train_set_size = [10, 20, 50, 100, 200, 500, 1000]
        self.reset_val()

        self.distance = distance
        self.stride = stride
        self.embed_scale = 0.1
        self.build_models()

    def reset_val(self):
        self.val_metrics = {}
        for n in self.val_train_set_size:
            self.val_metrics[f'val_arand_{n}'] = []
            self.val_metrics[f'val_seg_{n}'] = []


    @staticmethod
    def add_model_specific_args(parser):
        # parser.add_argument('--n_workers', type=int, default=10)
        parser.add_argument('--head_layers', default=2, type=int)
        parser.add_argument('--encoder_layers', default=2, type=int)
        parser.add_argument('--ndim', type=int, default=2)
        parser.add_argument('--hidden_channels', type=int, default=64)
        parser.add_argument('--distance', type=int, default=8)
        parser.add_argument('--in_channels', type=int, default=1)
        parser.add_argument('--out_channels', type=int, default=3)
        parser.add_argument('--initial_lr', type=float, default=1e-4)
        parser.add_argument('--loss_name', type=str, default="CPC")
        parser.add_argument('--unet_type', type=str, default="gp")
        parser.add_argument('--patchsize', type=int, default=16)
        parser.add_argument('--patchoverlap', type=int, default=5)
        parser.add_argument('--patchdilation', type=int, default=1)

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

        if self.unet_type == "gp":
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
        elif self.unet_type == "dunet":
            self.unet = DUNet(self.in_channels, 16, N=8, return_hypercolumns=True)
        elif self.unet_type == "deeplab":
            self.unet = Deeplabv3(self.in_channels, self.hidden_channels)
        elif self.unet_type == "resnet":
            self.unet = PatchedResnet50(self.in_channels,
                                        self.hidden_channels,
                                        patchsize=self.patchsize,
                                        overlap=self.patchoverlap,
                                        dilation=self.patchdilation)
        else:
            raise NotImplementedError(f"model type {self.unet_type} not found")

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

    # def validation_end(self, outputs):
    #     avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
    #     tensorboard_logs = {'val_loss': avg_loss}
    #     return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.initial_lr)
        scheduler = MultiStepLR(optimizer, milestones=[500], gamma=0.1)
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
                              tuple(vis(self.squeeze(y_0[c]))
                                    for c in range(y_0.shape[0]))
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

        if self.stride > 0:
            xrshift = random.randint(0, self.stride)
            yrshift = random.randint(0, self.stride)

            strided_embedding = embedding[...,
                                          xrshift::self.stride, yrshift::self.stride]
        else:
            strided_embedding = embedding

        preds = self.prediction_cnn(strided_embedding)
        targets = self.target_cnn(strided_embedding)
        targets = strided_embedding

        b, c, h, w = targets.shape
        # (b, c, h, w) -> (num_vectors, emb_dim)
        # every vector (c-dim) is a target
        targets_perm = targets.permute(
            0, 2, 3, 1).contiguous().reshape([-1, c])

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

    def loss_CPCshift(self, y, embedding, offsets):


        if self.stride > 0:
            xrshift = random.randint(0, self.stride)
            yrshift = random.randint(0, self.stride)
            strided_embedding = embedding[...,
                                          xrshift::self.stride, yrshift::self.stride]
        else:
            strided_embedding = embedding

        preds = self.prediction_cnn(strided_embedding)
        # preds = strided_embedding
        targets = self.target_cnn(strided_embedding)
        # targets = strided_embedding.detach()

        b, c, h, w = targets.shape
        # (b, c, h, w) -> (num_vectors, emb_dim)
        # every vector (c-dim) is a target
        targets_perm = targets.permute(
            0, 2, 3, 1).contiguous().reshape([-1, c])

        # select the future (south) targets to predict
        # selects all of the ones south of the current source
        preds_i = preds * self.embed_scale

        # (b, c, h, w) -> (b*w*h, c) (all features)
        # this ordering matches the targets
        preds_i = preds_i.permute(
            0, 2, 3, 1).contiguous().reshape([-1, c])

        # calculate the strength scores
        logits = torch.matmul(preds_i, targets_perm.transpose(-1, -2))

        # generate the labels
        n = b * h * w

        # batch number
        b1 = (torch.arange(n) // (h * w)).long()
        # column number
        c1 = ((torch.arange(n) // w) % (h)).long()
        # row number
        r1 = (torch.arange(n) % w).long()

        x_offset, y_offset = offsets

        coordinate = ((b1 * h + c1) * w) + r1
        labels = coordinate + x_offset + (y_offset * w)

        # check that coordinate + x_offset is within bounds
        labels[r1 + x_offset < 0] = -100
        labels[r1 + x_offset >= w] = -100
        # check that coordinate + y_offset is within bounds
        labels[c1 + y_offset < 0] = -100
        labels[c1 + y_offset >= h] = -100

        # # check that coordinate + x_offset is within bounds
        # labels[r1 - abs(x_offset) < 0] = -100
        # labels[r1 + abs(x_offset) >= w] = -100
        # # check that coordinate + y_offset is within bounds
        # labels[c1 - abs(y_offset) < 0] = -100
        # labels[c1 + abs(y_offset) >= h] = -100

        labels = labels.to(logits.device)
        loss = nn.functional.cross_entropy(logits, labels, ignore_index=-100)

        return loss, targets[:, :3], preds[:, :3]

    def loss_CPCrandomshift(self, y, embedding):

        angle = 2 * np.pi * np.random.random()
        x_offset = int(self.distance * np.sin(angle))
        y_offset = int(self.distance * np.cos(angle))
        offsets = (x_offset, y_offset)

        return self.loss_CPCshift(y, embedding, offsets)

    def loss_CPCfixedshift(self, y, embedding):

        x_offset = 0
        y_offset = self.distance
        offsets = (x_offset, y_offset)

        return self.loss_CPCshift(y, embedding, offsets)

    def loss_stardist(self, y, embedding, flipped_embedding):

        predicted_star_distances = torch.nn.functional.softplus(
            self.star_head(embedding))

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

    def loss_spatialconsensus(self, y, embedding):

        angle = 2 * np.pi * np.random.random()
        x_offset = int(self.distance * np.sin(angle))
        y_offset = int(self.distance * np.cos(angle))
        
        shift = embedding.new([[x_offset, y_offset , 0]])[..., None, None]
        offset = (x_offset, y_offset)

        e = embedding[offset_slice(offset, reverse=False, extra_dims=2)]
        eshifted = embedding[offset_slice(offset, reverse=True, extra_dims=2)]
        eshifted = e + shift

        loss = ((e - eshifted)**2).mean()
        loss = loss + 0.000001 * embedding.mean()

        return loss, e, eshifted




class ContextPredictionTrainer(SSLTrainer):


    def build_models(self):

        super().build_models()
        
        # Todo: Could be exposed as parameter
        self.max_directions = 8

        self.context_mlp = MLP(2*self.hidden_channels, self.max_directions,
                              n_hidden_layers=self.encoder_layers,
                              n_hidden=self.hidden_channels, ndim=2)

    def loss_fn(self):
        return getattr(self, "loss_" + self.loss_name)

    def loss_directionclass(self, x, y):
        c_split = x.shape[-3] // 2
        assert(c_split > 0)

        embedding = self.forward(x[..., :c_split, :, :])
        embedding_shift = self.forward(x[..., c_split:, :, :])

        stacked_emb = torch.cat((embedding, embedding_shift), dim=1)
        context_pred = self.context_mlp(stacked_emb)

        expand_y = y[:, 0, None, None].long().expand(y.shape[0], context_pred.shape[-2], context_pred.shape[-1])
        loss = nn.functional.cross_entropy(context_pred, expand_y, ignore_index=-100)

        return loss, embedding, embedding_shift, context_pred, expand_y

    def loss_anchor(self, x, y):

        print("anchor loss!!!!!!", y)
        c_split = x.shape[-3] // 2
        assert(c_split > 0)

        embedding = self.forward(x[..., :c_split, :, :])
        embedding_shift = self.forward(x[..., c_split:, :, :])

        shift_offset = torch.zeros(embedding_shift.shape[:2] + (1, 1))
        shift_offset[:, :2] = y[:, 1:3, None, None] / 10.
        shift_offset = shift_offset.to(embedding_shift.device)

        anchor_shift = embedding_shift - shift_offset
        
        loss = torch.nn.functional.mse_loss(embedding,
                                            anchor_shift,
                                            reduction='mean')

        return loss, embedding, embedding_shift, None, None

    def training_step(self, batch, batch_nb):
        x, y = batch

        loss, embedding, embedding_shift, context_pred, expand_y = self.loss_fn()(x, y)

        tensorboard_logs = {'train_loss': loss.detach().cpu()}
        tensorboard_logs['iteration'] = self.global_step

        if self.log_now():

            c_split = x.shape[-3] // 2
            assert(c_split > 0)

            embedding_shift = embedding_shift.detach().cpu()
            embedding = embedding.detach().cpu()
            
            self.logger.experiment.add_histogram(
                "target_hist", y.detach().cpu(), self.global_step)
            self.logger.experiment.add_histogram(
                "embedding_hist", embedding[..., ::3, ::11, ::11].detach().cpu(), self.global_step)
            
            embedding_batch = torch.cat(tuple(torch.cat(tuple(vis(self.squeeze(torch.cat((e_0[c], e_s[c]), dim=-1))) for c in range(
                e_0.shape[0])), dim=1) for e_0, e_s in zip(embedding, embedding_shift)), dim=-1)
            # embedding_batch_shift = torch.cat(tuple(torch.cat(tuple(vis(self.squeeze(e_0[c])) for c in range(
            #     e_0.shape[0])), dim=1) for e_0 in embedding_shift.detach().cpu()), dim=-1)
            # embedding_batch = torch.cat((embedding_batch, embedding_batch_shift), dim=-1)

            self.logger.experiment.add_image(f'embedding', embedding_batch, self.global_step)

            if len(x.shape) == 4:
                img_batch = torch.cat(tuple(torch.cat(tuple(vis(self.squeeze(torch.cat((x_0[c], x_s[c]), dim=-1))) for c in range(
                    x_0.shape[0])), dim=1) for x_0, x_s in zip(x[:, :c_split].detach().cpu(), x[:, c_split:].detach().cpu())), dim=-1)
                self.logger.experiment.add_image(f'img', img_batch, self.global_step)
            # embedding_batch_shift = torch.cat(tuple(torch.cat(tuple(vis(self.squeeze(e_0[c])) for c in range(
            #     e_0.shape[0])), dim=1) for e_0 in embedding_shift.detach().cpu()), dim=-1)
            # embedding_batch = torch.cat((embedding_batch, embedding_batch_shift), dim=-1)

            if self.loss_name == "directionclass":
                vis_labels = tuple(_ for _ in context_pred.detach().cpu().argmax(dim=1).numpy())
                vis_labels = np.concatenate(vis_labels, axis=-1)
                vis_gtlabels = tuple(_ for _ in expand_y.detach().cpu().numpy())
                vis_gtlabels = np.concatenate(vis_gtlabels, axis=-1)
                vis_labels = label2color(np.concatenate((vis_labels, vis_gtlabels), axis=-2))

                self.logger.experiment.add_image(f'argmax_prediction', vis_labels, self.global_step)

                vis_probs = tuple(context_pred[b].detach().cpu().softmax(dim=0)[y[b]].numpy() for b in range(context_pred.shape[0]))
                vis_probs = np.concatenate(vis_labels, axis=-1)
                self.logger.experiment.add_image(f'vis_probs', vis_probs[None], self.global_step)

        return {'loss': loss, 'log': tensorboard_logs}

    def log_img(self, name, img, eval_directory=None):
        self.logger.experiment.add_image(
            name, img, self.global_step)
        if eval_directory is not None:
            try:
                if img.shape[0] == 1:
                    img = img[0]
                if img.shape[0] in [3, 4]:
                    img = img.transpose(1, 2, 0)
                print(os.path.join(eval_directory, name+".png"))
                print(name, img.min(), img.max())
                if img.max() <= 1. and img.min() >= 0.:
                    imsave(os.path.join(eval_directory, name+".png"),
                           (img*255.).astype(np.uint8),
                           check_contrast=False)
                else:
                    imsave(os.path.join(eval_directory, name+".png"),
                           img, check_contrast=False)
            except Exception as e:
                print(e)
                print("can not imsave ", img.shape)

    def validation_step(self, batch, batch_nb):
        eval_directory = os.path.abspath(os.path.join(self.logger.log_dir,
                                                      os.pardir,
                                                      os.pardir,
                                                      "evaluation",
                                                      f"{self.global_step:08d}"))
        
        x, y, labels_torch = batch

        labels = labels_torch[:, 0].cpu().numpy()
        del labels_torch

        # save model with evaluation number
        model_save_path = os.path.join(eval_directory, "model.torch")
        score_filename = os.path.join(eval_directory, "score.csv")

        os.makedirs(eval_directory, exist_ok=True)
        torch.save(self.unet.state_dict(), model_save_path)
        self.log_img(f'img_val', vis(x[-1, 0].detach().cpu().numpy()), eval_directory=eval_directory)

        with torch.no_grad():
            with open(score_filename, "w") as scf:


                c_split = x.shape[1] // 2
                assert(c_split > 0)

                # embedding = self.forward(x[:, :c_split])
                # TODO: expose padding parameters
                embedding_torch = self.forward(x[:, :c_split])
                if self.unet_type == "resnet":
                    embedding_torch = self.unet.predict(x[:, :c_split], device="cpu")

                elif embedding_torch.shape[-1] != 256:
                    embedding_torch = self.forward(F.pad(x[:, :c_split], (72, 72, 72, 72), mode='reflect'))[:, :, 4:-4, 4:-4]

                embedding = embedding_torch.detach().cpu().numpy()                
                del embedding_torch
                del x
                del y

                embedding = np.transpose(embedding, (1,0,2,3))
                # self.log_img(f'embedding_0_2', embedding[:3, -1], eval_directory=eval_directory)
                C, _, W, H = embedding.shape

                pca_in = embedding[:, -1].reshape(C, -1).T
                pca = PCA(n_components=3, whiten=True)
                pca_out = pca.fit_transform(pca_in).T
                pca_image = pca_out.reshape(3, W, H)
                
                self.log_img(f'PCA', pca_image, eval_directory=eval_directory)
                self.log_img(f'gt_val', label2color(labels[-1].astype(np.int32)), eval_directory=eval_directory)

                # embedding_shift = self.forward(x[:, c_split:])

                # stacked_emb = torch.cat((embedding, embedding_shift), dim=1)
                # context_pred = self.context_mlp(stacked_emb)

                # expand_y = y[:, None, None].expand(y.shape[0], context_pred.shape[-2], context_pred.shape[-1])
                # loss = nn.functional.cross_entropy(context_pred, expand_y, ignore_index=-100)

                tensorboard_logs = {}#'val_loss': loss.detach()}
                tensorboard_logs['iteration'] = self.global_step

                train_stardist = np.stack(
                    [stardist.geometry.star_dist(l, n_rays=32) for l in labels])
                background = labels == 0
                inner = train_stardist.min(axis=-1) > 3
                # classes 0: boundaries, 1: inner_cell, 2: background
                threeclass = (2 * background) + inner

                fg_mask = labels > 0
                bg_mask = labels == 0

                dist_transform = \
                    np.stack(ndimage.morphology.distance_transform_edt(_) for _ in bg_mask)
                dist_mask = np.exp( - dist_transform / 8) > np.random.rand(*bg_mask.shape)
                bg_train_mask = np.logical_and(bg_mask, dist_mask)

                self.log_img(f'dist_mask', dist_mask[None, -1], eval_directory=eval_directory)
                self.log_img(f'bg_train_mask', bg_train_mask[None, -1], eval_directory=eval_directory)

                bg_indices = list(zip(*np.where(bg_train_mask)))
                c0_indices = list(zip(*np.where(threeclass == 0)))
                c1_indices = list(zip(*np.where(threeclass == 1)))

                random.Random(3).shuffle(c0_indices)
                random.Random(4).shuffle(c1_indices)
                random.Random(5).shuffle(bg_indices)

                for n_samples in self.val_train_set_size:

                    train_mask = np.zeros(fg_mask.shape, dtype=np.bool)
                    # shuffle is identical in every run
                    for b0, x0, y0 in itertools.islice(c0_indices, 0, n_samples):
                        train_mask[b0, x0, y0] = 1

                    for b0, x0, y0 in itertools.islice(c1_indices, 0, n_samples):
                        train_mask[b0, x0, y0] = 1

                    # shuffle is identical in every run
                    for b0, x0, y0 in itertools.islice(bg_indices, 0, n_samples):
                        train_mask[b0, x0, y0] = 1

                    training_data = embedding[:, train_mask].T
                    train_labels = threeclass[train_mask]

                    # foreground background
                    knn = KNeighborsClassifier(n_neighbors=3,
                                               weights='distance',
                                               n_jobs=-1)

                    knn.fit(training_data, train_labels)

                    spatial_dims = embedding.shape[1:]

                    flatt_embedding = np.transpose(
                        embedding.reshape((embedding.shape[0], -1)), (1, 0))
                    prediction = knn.predict(flatt_embedding)
                    prediction = prediction.reshape(spatial_dims)

                    inner = prediction == 1
                    background = prediction == 2

                    predicted_seg = np.stack([compute_3class_segmentation(
                        i, b) for i, b in zip(inner, background)])

                    arand_score = np.mean([adapted_rand(p, g)
                                         for p, g in zip(predicted_seg, labels)])
                    seg_score = np.mean([seg_metric(p, g)
                                         for p, g in zip(predicted_seg, labels)])

                    scf.write(f"{n_samples},{seg_score},{arand_score}\n")

                    self.log_img(f'3class_val_{n_samples}',
                            np.stack((prediction[-1] == 0, prediction[-1] == 1, prediction[-1] == 2),
                                     axis=0).astype(np.float32),
                            eval_directory=eval_directory)

                    self.log_img(f'instances_val_{n_samples}', label2color(predicted_seg[-1]), eval_directory=eval_directory)

                    tensorboard_logs[f"arand_n={n_samples}"] = arand_score
                    tensorboard_logs[f"seg_n={n_samples}"] = seg_score

                    self.val_metrics[f'val_arand_{n_samples}'].append(arand_score)
                    self.val_metrics[f'val_seg_{n_samples}'].append(seg_score)

        return tensorboard_logs

    def validation_epoch_end(self, outs):
        metrics = dict((k, np.mean(self.val_metrics[k])) for k in self.val_metrics)
        self.logger.log_metrics(metrics, step=self.global_step)
        self.reset_val()

    def loss_context_prediction(self, y, embedding):

        direction = random.randint(0, 8)
        x_offset, y_offset = offset_from_direction
        
        shift = embedding.new([[x_offset, y_offset , 0]])[..., None, None]
        offset = (x_offset, y_offset)

        e = embedding[offset_slice(offset, reverse=False, extra_dims=2)]
        eshifted = embedding[offset_slice(offset, reverse=True, extra_dims=2)]
        eshifted = e + shift

        loss = ((e - eshifted)**2).mean()
        loss = loss + 0.000001 * embedding.mean()

        return loss, e, eshifted
