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
from lisl.pl.model import MLP, get_unet_kernels, PatchedResnet50
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
                       hidden_channels=64,
                       in_channels=1,
                       out_channels=3,
                       stride=2,
                       initial_lr=1e-4,
                       regularization=1e-4,
                       anchor_radius=20,
                       lr_milestones=(100)):
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
        self.lr_milestones = list(int(_) for _ in lr_milestones)
        self.regularization = regularization
        self.anchor_radius = anchor_radius

        self.val_train_set_size = [10, 20, 50, 100, 200, 500, 1000]
        self.reset_val()

        self.distance = distance
        self.stride = stride
        self.embed_scale = 0.1
        self.build_models()
        self.build_loss()

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
        parser.add_argument('--hidden_channels', type=int, default=64)
        parser.add_argument('--distance', type=int, default=8)
        parser.add_argument('--in_channels', type=int, default=1)
        parser.add_argument('--out_channels', type=int, default=3)
        parser.add_argument('--initial_lr', type=float, default=1e-4)
        parser.add_argument('--regularization', type=float, default=1e-4)
        parser.add_argument('--loss_name', type=str, default="CPC")
        parser.add_argument('--unet_type', type=str, default="gp")
        parser.add_argument('--lr_milestones', nargs='*', default=[200])
        parser.add_argument('--anchor_radius', type=float, default=10)

        return parser

    def forward(self, x):
        return self.model(x)

    def forward_patches(self, x):
        # expects input in the shape of
        # x.shape = 
        # (batch, patch_size, channels, patch_width, patch_height)
        b, p, c, pw, ph = x.shape
        out = self.model(x.view(-1, c, pw, ph))
        return out.view(b, p, -1)

    def build_models(self):

        self.model = PatchedResnet50(self.in_channels,
                                    self.hidden_channels)

    def build_loss(self, ):
        self.anchor_loss = AnchorLoss(self.anchor_radius)

    def training_step(self, batch, batch_nb):
        x, patches, abs_coords, patch_matches, mask = batch

        embedding = self.forward_patches(patches)

        if self.global_step % 1000 == 0:

            # save model
            model_directory = os.path.abspath(os.path.join(self.logger.log_dir,
                                                          os.pardir,
                                                          os.pardir,
                                                          "model"))
            model_save_path = os.path.join(model_directory, f"model_{self.global_step}.torch")
            os.makedirs(model_directory, exist_ok=True)
            torch.save(self.model.state_dict(), model_save_path)

            if embedding.requires_grad:
                def log_hook(grad_input):
                    img_directory = os.path.abspath(os.path.join(self.logger.log_dir,
                                              os.pardir,
                                              os.pardir,
                                              "img"))
                    os.makedirs(img_directory, exist_ok=True)
                    for b in range(len(embedding)):
                        vis_anchor_embedding(embedding[b].detach().cpu().numpy(),
                            abs_coords[b].detach().cpu().numpy(),
                            x[b].detach().cpu().numpy(),
                            # grad=-grad_input[b].detach().cpu().numpy(),
                            output_file=f"{img_directory}/vis_{self.global_step}_{b}.png")
                    handle.remove()

                handle = embedding.register_hook(log_hook)

        # detach masked  boundary patches
        embedding = ((mask[..., None]).float() * embedding.detach()) + ((~mask[..., None]).float() * embedding)

        anchor_loss = self.anchor_loss(embedding, abs_coords, patch_matches)
        self.log('anchor_loss', anchor_loss.detach(), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        reg_loss = self.regularization * embedding.norm(2, dim=-1).sum()
        self.log('reg_loss', reg_loss.detach(), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        loss = anchor_loss + reg_loss

        # loss, y, y_hat = self.loss_anchor(y, embedding)
        tensorboard_logs = {'train_loss': loss.detach()}
        tensorboard_logs['iteration'] = self.global_step

        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        x, patches, abs_coords, patch_matches, mask, y = batch

        embedding = self.forward_patches(patches)
        loss = self.anchor_loss(embedding, abs_coords, patch_matches)

        # if self.log_now(val=True):
        #     self.log_batch_images(x, y, y_hat, embedding, "val")

        return {'val_loss': loss}

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.initial_lr, weight_decay=0.01)
        scheduler = MultiStepLR(optimizer, milestones=self.lr_milestones, gamma=0.1)
        return [optimizer], [scheduler]


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


# class ContextPredictionTrainer(SSLTrainer):


#     def build_models(self):

#         super().build_models()
        
#         # Todo: Could be exposed as parameter
#         self.max_directions = 8

#         self.context_mlp = MLP(2*self.hidden_channels, self.max_directions,
#                                n_hidden_layers=self.encoder_layers,
#                                n_hidden=self.hidden_channels, ndim=2)

#     def loss_fn(self):
#         return getattr(self, "loss_" + self.loss_name)

#     def loss_directionclass(self, x, y):
#         c_split = x.shape[-3] // 2
#         assert(c_split > 0)

#         embedding = self.forward(x[..., :c_split, :, :])
#         embedding_shift = self.forward(x[..., c_split:, :, :])

#         stacked_emb = torch.cat((embedding, embedding_shift), dim=1)
#         context_pred = self.context_mlp(stacked_emb)

#         expand_y = y[:, 0, None, None].long().expand(y.shape[0], context_pred.shape[-2], context_pred.shape[-1])
#         loss = nn.functional.cross_entropy(context_pred, expand_y, ignore_index=-100)

#         return loss, embedding, embedding_shift, context_pred, expand_y

#     def flip_forward_flip(self, x):

#         # x.shape = (b * n, c, s_1, ..., s_d)

#         flip_x, flip_y, flip_dims = random.choice([
#              (-1, 1, [-2]),
#              (1, -1, [-1]),
#              (-1, -1, [-1, -2])])

#         x = torch.flip(x, flip_dims)
        
#         # predict anchor embedding
#         e = self.forward(x.detach())
#         # e.shape = (b * n, d)

#         # flip the predicted tensors back
#         e[:, 0] *= flip_x
#         e[:, 1] *= flip_y
#         # Todo: remove if above works
#         # e = torch.cat((e[:, 0:1] * flip_x,
#         #                e[:, 1:2] * flip_y,
#         #                e[:, 2:]), dim=1)

#         return e



#     def loss_stardist(self, x, y):

#         c_split = x.shape[-3] // 2
#         assert(c_split > 0)

#         embedding = self.forward(x[..., :c_split, :, :])
#         embedding_shift = self.forward(x[..., c_split:, :, :])

        
#         # direction = random.randint(0, 8)
#         # x_offset, y_offset = offset_from_direction
        
#         # shift = embedding.new([[x_offset, y_offset , 0]])[..., None, None]
#         # offset = (x_offset, y_offset)

#         # e = embedding[offset_slice(offset, reverse=False, extra_dims=2)]
#         # eshifted = embedding[offset_slice(offset, reverse=True, extra_dims=2)]
#         # eshifted = e + shift

#         # loss = ((e - eshifted)**2).mean()
#         # loss = loss + 0.000001 * embedding.mean()

#         # return loss, e, eshifted


#     def training_step(self, batch, batch_nb):
#         x, y = batch

#         loss, embedding, embedding_shift, context_pred, expand_y = self.loss_fn()(x, y)

#         tensorboard_logs = {'train_loss': loss.detach().cpu()}
#         tensorboard_logs['iteration'] = self.global_step


#         if self.log_now():

#             c_split = x.shape[-3] // 2
#             assert(c_split > 0)

#             embedding_shift = embedding_shift.detach().cpu()
#             embedding = embedding.detach().cpu()
            
#             self.logger.experiment.add_histogram(
#                 "target_hist", y.detach().cpu(), self.global_step)
#             self.logger.experiment.add_histogram(
#                 "embedding_hist", embedding[..., ::3, ::11, ::11].detach().cpu(), self.global_step)
            
#             embedding_batch = torch.cat(tuple(torch.cat(tuple(vis(self.squeeze(torch.cat((e_0[c], e_s[c]), dim=-1))) for c in range(
#                 e_0.shape[0])), dim=1) for e_0, e_s in zip(embedding, embedding_shift)), dim=-1)
#             # embedding_batch_shift = torch.cat(tuple(torch.cat(tuple(vis(self.squeeze(e_0[c])) for c in range(
#             #     e_0.shape[0])), dim=1) for e_0 in embedding_shift.detach().cpu()), dim=-1)
#             # embedding_batch = torch.cat((embedding_batch, embedding_batch_shift), dim=-1)

#             self.logger.experiment.add_image(f'embedding', embedding_batch, self.global_step)

#             if len(x.shape) == 4:
#                 img_batch = torch.cat(tuple(torch.cat(tuple(vis(self.squeeze(torch.cat((x_0[c], x_s[c]), dim=-1))) for c in range(
#                     x_0.shape[0])), dim=1) for x_0, x_s in zip(x[:, :c_split].detach().cpu(), x[:, c_split:].detach().cpu())), dim=-1)
#                 self.logger.experiment.add_image(f'img', img_batch, self.global_step)
#             # embedding_batch_shift = torch.cat(tuple(torch.cat(tuple(vis(self.squeeze(e_0[c])) for c in range(
#             #     e_0.shape[0])), dim=1) for e_0 in embedding_shift.detach().cpu()), dim=-1)
#             # embedding_batch = torch.cat((embedding_batch, embedding_batch_shift), dim=-1)
#             elif len(x.shape) == 5:
#                 # assuming a patchified image for now

#                 print(x[0].shape)

#                 # assumes image is square
#                 w = int(math.sqrt(x.shape[1]))
#                 tiled_x = x.reshape(x.shape[:1] + (w, w) + x.shape[2:]).detach().cpu()

#                 tiled_x = tiled_x.permute(0, 3, 1, -2, 2, -1)

#                 wl = w * x.shape[-2]
#                 tiled_x = tiled_x.reshape(tiled_x.shape[:2] + (wl, wl))
#                 self.logger.experiment.add_image(f'input', torch.cat(tuple(
#                     torch.cat(tuple(visnorm(tt) for tt in t), dim=-2) for t in tiled_x),dim=-1)[None], self.global_step)


#             if self.loss_name == "directionclass":
#                 vis_labels = tuple(_ for _ in context_pred.detach().cpu().argmax(dim=1).numpy())
#                 vis_labels = np.concatenate(vis_labels, axis=-1)
#                 vis_gtlabels = tuple(_ for _ in expand_y.detach().cpu().numpy())
#                 vis_gtlabels = np.concatenate(vis_gtlabels, axis=-1)
#                 vis_labels = label2color(np.concatenate((vis_labels, vis_gtlabels), axis=-2))

#                 self.logger.experiment.add_image(f'argmax_prediction', vis_labels, self.global_step)

#                 vis_probs = tuple(context_pred[b].detach().cpu().softmax(dim=0)[y[b, 0].cpu().numpy()].numpy() for b in range(context_pred.shape[0]))
#                 vis_probs = np.concatenate(vis_labels, axis=-1)
#                 self.logger.experiment.add_image(f'vis_probs', vis_probs[None], self.global_step)

#         return {'loss': loss, 'log': tensorboard_logs}

#     def log_img(self, name, img, eval_directory=None):
#         self.logger.experiment.add_image(
#             name, img, self.global_step)
#         if eval_directory is not None:
#             try:
#                 if img.shape[0] == 1:
#                     img = img[0]
#                 if img.shape[0] in [3, 4]:
#                     img = img.transpose(1, 2, 0)
#                 print(os.path.join(eval_directory, name+".png"))
#                 print(name, img.min(), img.max())
#                 if img.max() <= 1. and img.min() >= 0.:
#                     imsave(os.path.join(eval_directory, name+".png"),
#                            (img*255.).astype(np.uint8),
#                            check_contrast=False)
#                 else:
#                     imsave(os.path.join(eval_directory, name+".png"),
#                            img, check_contrast=False)
#             except Exception as e:
#                 print(e)
#                 print("can not imsave ", img.shape)

#     def validation_step(self, batch, batch_nb):
#         eval_directory = os.path.abspath(os.path.join(self.logger.log_dir,
#                                                       os.pardir,
#                                                       os.pardir,
#                                                       "evaluation",
#                                                       f"{self.global_step:08d}"))
        
#         x, y, labels_torch = batch
#         labels = labels_torch[:, 0].cpu().numpy()
#         del labels_torch

#         val_image_counter = len(self.val_metrics[f'val_seg_{self.val_train_set_size[0]}'])

#         # save model with evaluation number
#         model_save_path = os.path.join(eval_directory, "model.torch")
#         score_filename = os.path.join(eval_directory, f"score_{val_image_counter}.csv")
#         embedding_filename = os.path.join(eval_directory, f"embedding_{val_image_counter}.h5")

#         os.makedirs(eval_directory, exist_ok=True)
#         torch.save(self.unet.state_dict(), model_save_path)
#         self.log_img(f'{val_image_counter}_img_val', vis(x[-1, 0].detach().cpu().numpy()), eval_directory=eval_directory)

#         with torch.no_grad():
#             with open(score_filename, "w") as scf:

#                 embedding_0, embedding_1 = self.predict(x)
#                 loss, _, _, _, _ = self.loss_fn()(x, y, 
#                             embedding_0=embedding_0,
#                             embedding_1=embedding_1)

#                 embedding = embedding_0.detach().cpu().numpy() 

#                 # free memory immediately
#                 loss = loss.detach()
#                 self.val_metrics[f'validation_loss'].append(loss)

#                 del embedding_0
#                 del embedding_1
#                 del x
#                 del y

#                 embedding = np.transpose(embedding, (1,0,2,3))
#                 # self.log_img(f'embedding_0_2', embedding[:3, -1], eval_directory=eval_directory)
#                 C, _, W, H = embedding.shape

#                 with h5py.File(embedding_filename, "w") as outf:
#                     outf.create_dataset("data", data=embedding, compression="lzf")

#                 for b in range(embedding.shape[1]):
#                     pca_in = embedding[:, b].reshape(C, -1).T
#                     pca = PCA(n_components=3, whiten=True)
#                     pca_out = pca.fit_transform(pca_in).T
#                     pca_image = pca_out.reshape(3, W, H)
#                     self.log_img(f'{val_image_counter}_{b}_PCA', vis(pca_image), eval_directory=eval_directory)
#                 for b in range(labels.shape[0]):
#                     self.log_img(f'{val_image_counter}_{b}_gt_val', label2color(labels[b].astype(np.int32)), eval_directory=eval_directory)

#                 tensorboard_logs = {'val_loss': loss}# loss.detach()}
#                 tensorboard_logs['iteration'] = self.global_step

#                 train_stardist = np.stack(
#                     [stardist.geometry.star_dist(l, n_rays=32) for l in labels])
#                 background = labels == 0
#                 inner = train_stardist.min(axis=-1) > 3
#                 # classes 0: boundaries, 1: inner_cell, 2: background
#                 threeclass = (2 * background) + inner

#                 fg_mask = labels > 0
#                 bg_mask = labels == 0

#                 dist_transform = \
#                     np.stack(ndimage.morphology.distance_transform_edt(_) for _ in bg_mask)
#                 dist_mask = np.exp( - dist_transform / 8) > np.random.rand(*bg_mask.shape)
#                 bg_train_mask = np.logical_and(bg_mask, dist_mask)

#                 for b in range(dist_mask.shape[0]):
#                     self.log_img(f'{val_image_counter}_{b}_dist_mask', dist_mask[None, b], eval_directory=eval_directory)
#                     self.log_img(f'{val_image_counter}_{b}_bg_train_mask', bg_train_mask[None, b], eval_directory=eval_directory)

#                 bg_indices = list(zip(*np.where(bg_train_mask)))
#                 c0_indices = list(zip(*np.where(threeclass == 0)))
#                 c1_indices = list(zip(*np.where(threeclass == 1)))

#                 random.Random(3).shuffle(c0_indices)
#                 random.Random(4).shuffle(c1_indices)
#                 random.Random(5).shuffle(bg_indices)

#                 for n_samples in self.val_train_set_size:

#                     train_mask = np.zeros(fg_mask.shape, dtype=np.bool)
#                     # shuffle is identical in every run
#                     for b0, x0, y0 in itertools.islice(c0_indices, 0, n_samples):
#                         train_mask[b0, x0, y0] = 1

#                     for b0, x0, y0 in itertools.islice(c1_indices, 0, n_samples):
#                         train_mask[b0, x0, y0] = 1

#                     # shuffle is identical in every run
#                     for b0, x0, y0 in itertools.islice(bg_indices, 0, n_samples):
#                         train_mask[b0, x0, y0] = 1

#                     training_data = embedding[:, train_mask].T
#                     train_labels = threeclass[train_mask]

#                     # foreground background
#                     knn = KNeighborsClassifier(n_neighbors=3,
#                                                weights='distance',
#                                                n_jobs=-1)

#                     knn.fit(training_data, train_labels)

#                     spatial_dims = embedding.shape[1:]

#                     flatt_embedding = np.transpose(
#                         embedding.reshape((embedding.shape[0], -1)), (1, 0))
#                     prediction = knn.predict(flatt_embedding)
#                     prediction = prediction.reshape(spatial_dims)

#                     inner = prediction == 1
#                     background = prediction == 2

#                     predicted_seg = np.stack([compute_3class_segmentation(
#                         i, b) for i, b in zip(inner, background)])

#                     arand_score = np.mean([adapted_rand(p, g)
#                                          for p, g in zip(predicted_seg, labels)])
#                     seg_score = np.mean([seg_metric(p, g)
#                                          for p, g in zip(predicted_seg, labels)])

#                     scf.write(f"{n_samples},{seg_score},{arand_score}\n")

#                     for b in range(predicted_seg.shape[0]):
#                         self.log_img(f'{val_image_counter}_{b}_3class_val_{n_samples}',
#                                 np.stack((prediction[b] == 0, prediction[b] == 1, prediction[b] == 2),
#                                          axis=0).astype(np.float32),
#                                 eval_directory=eval_directory)

#                         self.log_img(f'{val_image_counter}_{b}_instances_val_{n_samples}', label2color(predicted_seg[b]), eval_directory=eval_directory)

#                     tensorboard_logs[f"arand_n={n_samples}"] = arand_score
#                     tensorboard_logs[f"seg_n={n_samples}"] = seg_score

#                     self.val_metrics[f'val_arand_{n_samples}'].append(arand_score)
#                     self.val_metrics[f'val_seg_{n_samples}'].append(seg_score)
                    
#         return {'val_loss': loss, 'log': tensorboard_logs}

#     def validation_epoch_end(self, outs):
#         metrics = dict((k, np.mean(self.val_metrics[k])) for k in self.val_metrics)
#         self.logger.log_metrics(metrics, step=self.global_step)
#         self.reset_val()

