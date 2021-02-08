from lisl.predict.volume import predict_volume
from pytorch_lightning.callbacks import Callback
import argparse
import inspect
import torch
import os
import copy
from lisl.datasets import Dataset
import zarr
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import stardist
from skimage import measure
from scipy import ndimage
from skimage.morphology import watershed
from skimage.io import imsave
from lisl.pl.utils import adapted_rand, vis, label2color, try_remove, BuildFromArgparse
from ctcmetrics.seg import seg_metric
from sklearn.decomposition import PCA
from lisl.pl.utils import Patchify
from torch.nn import functional as F
from sklearn.cluster import MeanShift
from lisl.pl.visualizations import vis_anchor_embedding

from PIL import Image
import matplotlib


def compute_3class_segmentation(inner, background):
    seeds = measure.label(inner, background=0)
    distance = ndimage.distance_transform_edt(1 - background)
    return watershed(-distance, seeds, mask=1 - background)


class SupervisedLinearSegmentationValidation(Callback, BuildFromArgparse):

    def __init__(self, test_ds_filename, test_ds_name_raw, test_ds_name_seg, test_out_shape, test_input_name):
        self.test_filename = test_ds_filename
        self.test_ds_name_raw = test_ds_name_raw
        self.test_ds_name_seg = test_ds_name_seg

        self.test_out_shape = test_out_shape
        self.test_input_name = test_input_name

        super().__init__()

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=False)
        parser.add_argument('--test_ds_filename', type=str)
        parser.add_argument('--test_ds_name_raw', type=str)
        parser.add_argument('--test_ds_name_seg', type=str)
        parser.add_argument('--test_out_shape', nargs='*', default=(120, 120))
        parser.add_argument('--test_input_name', type=str, default="x")

        return parser

    def log_img_and_seg(self,
                        img,
                        label,
                        predictions,
                        postfix,
                        pl_module,
                        eval_directory,
                        predicted_labels=None,
                        score=None,
                        embeddings=None):

        def log_img(name, img):
            pl_module.logger.experiment.add_image(
                name, img, pl_module.global_step)
            try:
                if img.shape[0] == 1:
                    img = img[0]
                if img.shape[0] in [3, 4]:
                    img = img.transpose(1, 2, 0)
                imsave(os.path.join(eval_directory, name+".jpg"),
                       img,
                       check_contrast=False)
            except:
                print("can not imsave ", img.shape)

        log_img(f'img_{postfix}', vis(img))
        log_img(f'gt_{postfix}', label2color(label))
        log_img(f'3class_{postfix}',
                np.stack((predictions == 0, predictions == 1, predictions == 2),
                         axis=0).astype(np.float32))

        if predicted_labels is not None:
            log_img(f'instances_{postfix}', label2color(predicted_labels))

        if score is not None:
            pl_module.logger.log_metrics(score, step=pl_module.global_step)

        if embeddings is not None:
            _, C, W, H = embeddings.shape

            pca_in = embeddings[0].reshape(C, -1).T
            pca = PCA(n_components=3, whiten=True)
            pca_out = pca.fit_transform(pca_in).T
            pca_image = pca_out.reshape(3, W, H)

            log_img(f'PCA', pca_image)
            log_img(f'embedding_0_2', embeddings[0, :3])


    def train_simple_model(self, embeddings, target, pl_module, number_of_iterations=2001):

        model = MLP(embeddings.shape[1], 3).to(embeddings.device)
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        embeddings = embeddings.detach()

        optimizer.zero_grad()
        for i in tqdm(range(number_of_iterations), desc="training simple model "):
            train_predictions = model(embeddings)
            loss = F.cross_entropy(train_predictions, target)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print("final train loss ", loss)
        return model, train_predictions

    def load_train(self):
        train_img = Image.open(self.train_image).convert('RGB')
        train_label = Image.open(self.train_labels).convert('RGB')
        test_img = Image.open(self.test_image).convert('RGB')
        test_label = Image.open(self.test_labels).convert('RGB')

        return train_img, train_label, test_img, test_label

    def on_validation_epoch_end(self, trainer, pl_module):

        # save model with evaluation number
        eval_directory = os.path.abspath(os.path.join(pl_module.logger.log_dir,
                                                      os.pardir,
                                                      os.pardir,
                                                      "evaluation",
                                                      f"{pl_module.global_step:08d}"))
        model_save_path = os.path.join(eval_directory, "model.torch")
        prediction_filename = os.path.join(eval_directory, "embeddings.zarr")
        score_filename = os.path.join(eval_directory, "score.csv")
        try_remove(prediction_filename)

        os.makedirs(eval_directory, exist_ok=True)
        torch.save(trainer.model.unet.state_dict(), model_save_path)

        dataset = Dataset(self.test_filename, (self.test_ds_name_raw, ))

        with torch.no_grad():
            model = trainer.model.unet.eval()

            predict_volume(model,
                           dataset,
                           eval_directory,
                           "embeddings.zarr",
                           ("embedding", ),
                           input_name=self.test_input_name,
                           checkpoint=None,
                           normalize_factor='skip',
                           model_output=0,
                           in_shape=(256, 256),
                           out_shape=self.test_out_shape,
                           spawn_subprocess=False,
                           num_workers=0,
                           z_is_time=True)

        z_array = zarr.open(prediction_filename, mode="r+")
        z_train = zarr.open(self.test_filename)

        embeddings = z_array['embedding'][:]
        instance_number_array = z_train['samples'][:]
        # This array conatins consecutive integers
        #    (one sequence per class)
        #    (starting at zero)
        #    (randomly permutated)
        labels = z_train['seg'][:].astype(np.int)

        with open(score_filename, "w") as scf:

            for n_samples in [5, 10, 20, 50, 100, 200, 500, 1000]:

                train_stardist = np.stack(
                    [stardist.geometry.star_dist(l, n_rays=32) for l in labels])
                background = labels == 0
                inner = train_stardist.min(axis=-1) > 3
                # classes 0: boundaries, 1: inner_cell, 2: background
                threeclass = (2 * background) + inner

                train_mask = instance_number_array <= n_samples
                training_data = embeddings[:, train_mask].T
                train_labels = threeclass[train_mask]

                # foreground background
                knn = KNeighborsClassifier(n_neighbors=3,
                                           weights='distance',
                                           n_jobs=-1)

                knn.fit(training_data, train_labels)

                test_mask = np.logical_and(instance_number_array > n_samples,
                                           instance_number_array < instance_number_array.max())

                spatial_dims = embeddings.shape[1:]

                flatt_embeddings = np.transpose(
                    embeddings.reshape((embeddings.shape[0], -1)), (1, 0))
                prediction = knn.predict(flatt_embeddings)
                prediction = prediction.reshape(spatial_dims)

                inner = prediction == 1
                background = prediction == 2

                predicted_seg = np.stack([compute_3class_segmentation(
                    i, b) for i, b in zip(inner, background)])


                z_array.create_dataset(f"prediction_{n_samples:04d}", data=prediction, compression='gzip')
                z_array.create_dataset(f"predicted_seg_{n_samples:04d}", data=predicted_seg, compression='gzip')
                z_array.create_dataset(f"labels_{n_samples:04d}", data=labels, compression='gzip')

                arand_score = np.mean([adapted_rand(p, g)
                                     for p, g in zip(predicted_seg, labels)])
                seg_score = np.mean([seg_metric(p, g)
                                     for p, g in zip(predicted_seg, labels)])

                scf.write(f"{n_samples},{seg_score},{arand_score}\n")

                self.log_img_and_seg(z_train['raw'][-1],
                                     labels[-1],
                                     prediction[-1],
                                     f"test_n={n_samples:04d}",
                                     pl_module,
                                     eval_directory,
                                     score={f"arand_n={n_samples}": arand_score,
                                            f"seg_n={n_samples}": seg_score},
                                     predicted_labels=predicted_seg[-1],
                                     embeddings=embeddings)

class AnchorSegmentationValidation(Callback):

    def __init__(self, run_ms_segmentation=True, device='cpu'):
        self.run_ms_segmentation = run_ms_segmentation
        self.device = device

        super().__init__()

    def on_validation_epoch_start(self, trainer, pl_module):
        """Called when the validation loop begins."""
        self.seg_scores = []

    def on_validation_epoch_end(self, trainer, pl_module):
        """Called when the validation loop ends."""
        print("seg_score_best_bandwidth", np.mean(self.seg_scores))
        pl_module.log("seg_score_best_bandwidth", np.mean(self.seg_scores))


    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):

        edim = pl_module.out_channels

        with torch.no_grad():
            x, patches, abs_coords, patch_matches, mask, y = batch

            patch_size = trainer.datamodule.patch_size
            lp = (patch_size // 2)
            rp = ((patch_size - 1) // 2)

            # padd the image to get one patch corresponding to each pixel 
            padded = F.pad(x, (lp, rp, lp, rp), mode='reflect')

            pf = Patchify(patch_size=patch_size,
                          overlap_size=patch_size-1,
                          dilation=1)

            embedding_relative = torch.empty((x.shape[0], edim) + x.shape[-2:])
            embedding = torch.empty((x.shape[0], edim) + x.shape[-2:])
            for i in range(x.shape[-2]):
                patches = torch.stack(list(pf(x0) for x0 in padded[:, :, i:i+(patch_size)]))
                # patches.shape = (batch_size, num_patches, 2, patch_width, patch_height)
                b, p, c, pw, ph = patches.shape

                patches = patches.cuda()
                pred_i = pl_module.forward_patches(patches)
                pred_i = pred_i.to(embedding.device)

                embedding_relative[:, :, i] = pred_i.permute(0, 2, 1).view(b, edim, x.shape[-1])
                embedding[:, :, i] = pred_i.permute(0, 2, 1).view(b, edim, x.shape[-1])
                embedding[:, 0, i] += torch.arange(x.shape[-1])[None]
                embedding[:, 1, i] += i

        # save model with evaluation number
        eval_directory = os.path.abspath(os.path.join(pl_module.logger.log_dir,
                                                      os.pardir,
                                                      os.pardir,
                                                      "evaluation",
                                                      f"{pl_module.global_step:08d}"))

        os.makedirs(eval_directory, exist_ok=True)
        def n(a):
            out = a - a.min()
            out /= out.max() + 1e-8 
            return out

        for b, e in enumerate(embedding.cpu().numpy()):
            for c in range(0, embedding.shape[1], 2):
                imsave(f"{eval_directory}/embedding_{batch_idx}_{pl_module.local_rank}_{b}_{c}.jpg", np.stack((n(e[c]), n(x[b, 0].cpu().numpy()), n(e[c+1])), axis=-1))

        embedding_relative = embedding_relative.cpu().numpy()
        for b, e in enumerate(embedding_relative):

            z_array = zarr.open(f"{eval_directory}/embedding_{batch_idx}_{pl_module.local_rank}_{b}.zarr", mode="w")
            z_array.create_dataset(f"embedding", data=e, compression='gzip')
            z_array.create_dataset(f"raw", data=x[b].cpu().numpy(), compression='gzip')


            imsave(f"{eval_directory}/rel_embedding_{batch_idx}_{pl_module.local_rank}_{b}_{c}.jpg", np.stack((n(e[0]), n(x[b, 0].cpu().numpy()), n(e[1])), axis=-1))
            cx = np.arange(e.shape[-2], dtype=np.float32)
            cy = np.arange(e.shape[-1], dtype=np.float32)
            coords = np.meshgrid(cx, cy, copy=True)
            coords = np.stack(coords, axis=-1)
            
            e_transposed = np.transpose(e, (1,2,0))
            # downsample_factor
            dsf = 8
            spatial_dims = coords.shape[-1]
            patch_coords = coords[dsf//2::dsf, dsf//2::dsf].reshape(-1, spatial_dims)
            patch_embedding = e_transposed[dsf//2::dsf, dsf//2::dsf, :spatial_dims].reshape(-1, spatial_dims)

            vis_anchor_embedding(patch_embedding,
                                 patch_coords,
                                 n(x[b].cpu().numpy()),
                                 grad=None,
                                 output_file=[f"{eval_directory}/pointer_embedding_{batch_idx}_{pl_module.local_rank}_{b}.jpg",
                                              f"{eval_directory}/pointer_embedding_{batch_idx}_{pl_module.local_rank}_{b}.pdf"])

        if self.run_ms_segmentation:
            b, c, w, h = embedding.shape
            resh_emb = embedding.permute(0, 2, 3, 1).view(b, w*h, c)

            bandwidths = [1, 2, 4] + list(np.arange(5., 30., 10.))
            seg_score_table = np.zeros((len(bandwidths), b))
            for i, bandwidth in enumerate(bandwidths):
                for j in range(b):
                    X = resh_emb[j].contiguous().numpy()
                    ms = MeanShift(bandwidth=bandwidth)
                    ms_seg = ms.fit(X[np.random.rand(len(X)) < 0.02])
                    ms_seg = ms.predict(X)
                    ms_seg = ms_seg.reshape(w, h)
                    colseg = label2color(ms_seg).transpose(1, 2, 0)

                    seg_score_table[i, j] = seg_metric(ms_seg, y[0].numpy())

                    if i == 0 or seg_score_table[i, j] >= seg_score_table[:i, j].max():
                        img = np.repeat(x[j, 0, ..., None].cpu().numpy(), 3, axis=-1)
                        blend = (colseg[..., :3] / 2)  + img
                        imsave(f'{eval_directory}/msseg_{batch_idx}_{pl_module.local_rank}_{j}.jpg', blend)

            print("seg_score", seg_score_table)

            self.seg_scores.append(seg_score_table.max(axis=0).mean())
