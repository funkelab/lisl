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
import skimage
from embeddingutils.affinities import embedding_to_affinities
from affogato.segmentation import compute_mws_clustering, compute_mws_segmentation

from PIL import Image
import matplotlib
from skimage.measure import label as label_cont

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
        vt_save_path = os.path.join(eval_directory, "model_vt.torch")
        prediction_filename = os.path.join(eval_directory, "embeddings.zarr")
        score_filename = os.path.join(eval_directory, "score.csv")
        try_remove(prediction_filename)

        os.makedirs(eval_directory, exist_ok=True)
        torch.save(trainer.model.unet.state_dict(), model_save_path)
        torch.save(trainer.spatial_transformer.state_dict(), vt_save_path)
        

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

def n(a):
    out = a - a.min()
    out /= out.max() + 1e-8 
    return out

class AnchorSegmentationValidation(Callback):

    def __init__(self, run_segmentation=False, device='cpu'):
        self.run_segmentation = run_segmentation
        self.device = device

        super().__init__()

    def on_validation_epoch_start(self, trainer, pl_module):
        """Called when the validation loop begins."""
        self.seg_scores = {}

    def on_validation_epoch_end(self, trainer, pl_module):
        """Called when the validation loop ends."""
        for k in self.seg_scores:
            print("seg_score_best_bandwidth", np.mean(self.seg_scores[k]))
            pl_module.log(f"{k}_seg_score", np.mean(self.seg_scores[k]))

    def predict_embedding(self, batch, pl_module, patch_size):

        edim = pl_module.out_channels

        with torch.no_grad():
            x, patches, abs_coords, patch_matches, mask, y = batch

            lp = (patch_size // 2)
            rp = ((patch_size - 1) // 2)

            # padd the image to get one patch corresponding to each pixel 
            padded = F.pad(x, (lp, rp, lp, rp), mode='reflect')

            pf = Patchify(patch_size=patch_size,
                          overlap_size=patch_size-1,
                          dilation=1)

            embedding_relative = torch.empty((x.shape[0], edim) + x.shape[-2:])
            embedding = None
            
            for i in range(x.shape[-2]):
                patches = torch.stack(list(pf(x0) for x0 in padded[:, :, i:i+(patch_size)]))
                # patches.shape = (batch_size, num_patches, 2, patch_width, patch_height)
                b, p, c, pw, ph = patches.shape

                patches = patches.contiguous().cuda()
                coordinates = torch.stack((torch.arange(x.shape[-1]).float(),
                                           i * torch.ones((x.shape[-1], )).float()), dim=-1)[None]
                coordinates = coordinates.to(patches.device)

                pred_i, _ = pl_module.forward_patches(patches, coordinates)
                pred_i_abs = pl_module.anchor_loss.absoute_embedding(pred_i, coordinates)
                abs_edim = pred_i_abs.shape[-1]
                if embedding is None:
                    embedding = torch.empty((x.shape[0], abs_edim) + x.shape[-2:])

                pred_i = pred_i.to(embedding_relative.device)
                pred_i_abs = pred_i_abs.to(embedding.device)

                embedding_relative[:, :, i] = pred_i.permute(0, 2, 1).view(b, edim, x.shape[-1])
                embedding[:, :, i] = pred_i_abs.permute(0, 2, 1).view(b, abs_edim, x.shape[-1])

        return embedding, embedding_relative

    def create_eval_dir(self, pl_module):
        eval_directory = os.path.abspath(os.path.join(pl_module.logger.log_dir,
                                                      os.pardir,
                                                      os.pardir,
                                                      "evaluation",
                                                      f"{pl_module.global_step:08d}"))

        os.makedirs(eval_directory, exist_ok=True)
        return eval_directory


    def visualize_embeddings(self, embedding, x, filename):
        for b in range(len(embedding)):
            e = embedding[b].cpu().numpy()
            for c in range(0, embedding.shape[1], 2):
                imsave(f"{filename}_{b}_{c}.jpg", 
                       np.stack((n(e[c]), n(x[b, 0].cpu().numpy()), n(e[c+1])), axis=-1))

    def visualize_segmentation(self, seg, x, filename):
        colseg = label2color(seg).transpose(1, 2, 0)
        img = np.repeat(x[0, ..., None].cpu().numpy(), 3, axis=-1)
        blend = (colseg[..., :3] / 2)  + img
        imsave(filename, blend)


    def visualize_embedding_vectors(self, embedding_relative, x, filename, downsample_factor=8):

        for b in range(len(embedding_relative)):
            e = embedding_relative[b].cpu().numpy()

            cx = np.arange(e.shape[-2], dtype=np.float32)
            cy = np.arange(e.shape[-1], dtype=np.float32)
            coords = np.meshgrid(cx, cy, copy=True)
            coords = np.stack(coords, axis=-1)
            
            e_transposed = np.transpose(e, (1,2,0))

            dsf = downsample_factor
            spatial_dims = coords.shape[-1]
            patch_coords = coords[dsf//2::dsf, dsf//2::dsf].reshape(-1, spatial_dims)
            patch_embedding = e_transposed[dsf//2::dsf, dsf//2::dsf, :spatial_dims].reshape(-1, spatial_dims)

            vis_anchor_embedding(patch_embedding,
                                 patch_coords,
                                 n(x[b].cpu().numpy()),
                                 grad=None,
                                 output_file=f"{filename}_{b}.jpg")

    def write_embedding_to_file(self, eval_data_file, embedding, embedding_relative, x, y):
        # write embedding and raw data to file
        z_array = zarr.open(eval_data_file, mode="w")
        for b in range(len(embedding_relative)):
            z_array.create_dataset(f"{b}/embedding", data=embedding_relative[b].cpu().numpy(), compression='gzip')
            z_array.create_dataset(f"{b}/embedding_abs", data=embedding.cpu().numpy()[b], compression='gzip')
            z_array.create_dataset(f"{b}/gt_segmentation", data=y.cpu().numpy()[b, None], compression='gzip')
            z_array.create_dataset(f"{b}/raw", data=x[b].cpu().numpy(), compression='gzip')
            threshold = skimage.filters.threshold_li(image=x[b].cpu().numpy())
            z_array.create_dataset(f"{b}/threshold_li", data=255*(x[b].cpu().numpy() > threshold).astype(dtype=np.uint8), compression='gzip')

        return z_array

    def meanshift_segmentation(self, embedding, bandwidths):
        # Compute Meanshift Segmentation
        ms_segmentaiton = {}

        for i, bandwidth in enumerate(bandwidths):
            ms_segmentaiton[bandwidth] = []

            segment_with_meanshift = AnchorMeanshift(bandwidth)
            ms_segmentation = segment_with_meanshift(embedding)

            for b, seg in enumerate(ms_segmentation):
                ms_segmentaiton[bandwidth].append(seg)

        return ms_segmentaiton

    def mws_segmentation(self, embedding, temperatures, foreground_mask=None, img_dir_and_prefix=None, output_z_array=None):

        mws_segmentaiton = {}

        # compute affinities
        att_c = 2
        offsets = np.array([[-1, 0], [0, -1],
                            [-9, 0], [0, -9],
                            [-9, -9], [9, -9],
                            [-9, -4], [-4, -9], [4, -9], [9, -4],
                            [-27, 0], [0, -27]], int)

        bias_list = [0.]#np.arange(-0.5, 0.5, 0.1)
        
        for i, temperature in enumerate(temperatures):
            def affinity_measure(x, y, dim=0):
                distance = (x - y).norm(2, dim=dim)
                return (-distance.pow(2) / temperature).exp()

            for b, emb in enumerate(embedding):
                affinities = embedding_to_affinities(emb,
                                                     offsets=offsets,
                                                     affinity_measure=affinity_measure)

                if img_dir_and_prefix is not None:
                    img_dir, prefix = img_dir_and_prefix
                    for c in range(len(affinities)):
                        imsave(f"{img_dir}/affinity_{prefix}_{b}_{temperature}_{c}.jpg", 
                               affinities[c].cpu().numpy())
                if output_z_array is not None:
                    output_z_array.create_dataset(f"{b}/affinities_{temperature}",
                                           data=affinities.cpu().numpy(),
                                           compression='gzip', overwrite=True)
                # compute affinityies
                affinities[att_c:] *= -1
                affinities[att_c:] += 1

                for bias_value in bias_list:

                        bias = np.array([0,] * att_c + [bias_value, ] * (affinities.shape[0] - att_c))[:, None, None]
                        use_mask = foreground_mask is not None
                        mask = foreground_mask[b] if use_mask else None

                        seg = compute_mws_segmentation(affinities + bias,
                                           offsets,
                                           att_c,
                                           strides=[2, 2],
                                           mask=mask).astype(np.int32)

                        seg = label_cont(seg)
                        key = f"{temperature}_bias{bias_value}_fg{use_mask}"
                        if key not in mws_segmentaiton:
                            mws_segmentaiton[key] = []

                        mws_segmentaiton[key].append(seg)
            

        return mws_segmentaiton



    def log_SEG_score(self, segmentation_dict, gt_segmentation, evaluation_key=None):

        seg_score_table = []
        for k in segmentation_dict:
            seg_scores = []
            for j, seg in enumerate(segmentation_dict[k]):
                seg_scores.append(seg_metric(seg, gt_segmentation[j].numpy()))
            seg_score_table.append(seg_scores)

        seg_score_table = np.array(seg_score_table)
        SEG_score = seg_score_table.max(axis=0).mean()

        if evaluation_key is not None:
            if evaluation_key not in self.seg_scores:
                self.seg_scores[evaluation_key] = []
            self.seg_scores[evaluation_key].append(SEG_score)

        return SEG_score

    def write_segmentation_to_zarr(self, segmentation_dict, key, z_array):
        for k in segmentation_dict:
            for b, seg in enumerate(segmentation_dict[k]):
                z_array.create_dataset(f"{b}/{key}_{k}", data=label_cont(seg[None]), compression='gzip')

    def visualizalize_segmentation_dict(self, segmentation_dict, x, filename):
        for k in segmentation_dict:
            for b, seg in enumerate(segmentation_dict[k]):
                print(f'writing {filename}_{b}_{k}.jpg')
                self.visualize_segmentation(seg, x[b], f'{filename}_{b}_{k}.jpg')

    def full_evaluation(self, trainer, pl_module, batch, batch_idx, dataloader_idx):

        x, patches, abs_coords, patch_matches, mask, y = batch

        eval_directory = self.create_eval_dir(pl_module)

        embedding, embedding_relative = self.predict_embedding(batch,
                                                               pl_module,
                                                               trainer.datamodule.patch_size)
        
        vis_pointer_filename = f"{eval_directory}/pointer_embedding_{batch_idx}_{pl_module.local_rank}"
        vis_embedding_filename = f"{eval_directory}/embedding_{batch_idx}_{pl_module.local_rank}"
        vis_relembedding_filename = f"{eval_directory}/embedding_relative_{batch_idx}_{pl_module.local_rank}"
        self.visualize_embedding_vectors(embedding_relative, x, vis_pointer_filename)
        self.visualize_embeddings(embedding, x, vis_embedding_filename)
        self.visualize_embeddings(embedding_relative, x, vis_relembedding_filename)
        
        eval_data_file = f"{eval_directory}/embedding_{batch_idx}_{pl_module.local_rank}.zarr"
        z_array = self.write_embedding_to_file(eval_data_file, embedding, embedding_relative, x, y)

        if self.run_segmentation:

            ms_bandwidths = (1, 2, 4) + tuple(np.arange(5., 30., 10.))
            ms_seg = self.meanshift_segmentation(embedding, ms_bandwidths)

            evaluation_key = f"meanshift"
            seg_score = self.log_SEG_score(ms_seg, y, evaluation_key=evaluation_key)
            self.write_segmentation_to_zarr(ms_seg, "meanshift", z_array)
            self.visualizalize_segmentation_dict(ms_seg, x, f'{eval_directory}/meanshift_seg_{batch_idx}_{pl_module.local_rank}')

            temperatures = [1., 10., 100., 1000., 10000.]
            foreground_mask = [(np.array(z_array[f"{b}/threshold_li"]) > 0)[0] for b in range(len(embedding))]

            mws_seg = self.mws_segmentation(embedding, temperatures, foreground_mask=foreground_mask,
                                            img_dir_and_prefix=(eval_directory, f"{batch_idx}_{pl_module.local_rank}"))
            evaluation_key = f"MWS"
            seg_score = self.log_SEG_score(mws_seg, y, evaluation_key=f"MWS")
            self.write_segmentation_to_zarr(mws_seg, "MWS", z_array)
            self.visualizalize_segmentation_dict(mws_seg, x, f'{eval_directory}/mws_seg_{batch_idx}_{pl_module.local_rank}')

    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        self.full_evaluation(trainer, pl_module, batch, batch_idx, dataloader_idx)

class AnchorMeanshift():
    def __init__(self, bandwidth, reduction_probability = 0.05):
        self.ms = MeanShift(bandwidth=bandwidth)
        self.reduction_probability = reduction_probability

    def __call__(self, embedding):
        b, c, w, h = embedding.shape
        resh_emb = embedding.permute(0, 2, 3, 1).view(b, w*h, c)

        segmentation = []
        for j in range(b):
            X = resh_emb[j].contiguous().numpy()

            if self.reduction_probability < 1.:
                X_reduced = X[np.random.rand(len(X)) < self.reduction_probability]
                ms_seg = self.ms.fit(X_reduced)
            else:
                ms_seg = self.ms.fit(X)

            ms_seg = self.ms.predict(X)
            ms_seg = ms_seg.reshape(w, h)
            segmentation.append(ms_seg)

        return np.stack(segmentation)


if __name__ == '__main__':
    

    from EmbedSeg.utils.utils2 import matching_dataset, obtain_AP_one_hot
    import csv

    score_filename = "/nrs/funke/wolfs2/lisl/experiments/dense_run_01/01_train/dsb2_details.csv"

    with open(score_filename, 'w', newline='', buffering=1) as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)

        for i in [31]:#list(range(34, 38, 1)):
            for j in range(0, 15, 1):
                eval_file = f'/nrs/funke/wolfs2/lisl/experiments/dense_run_01/01_train/setup_t{i:04}/evaluation/noaugmentation/00040399/embedding_{j}_0.zarr'

    # score_filename = "/nrs/funke/wolfs2/lisl/experiments/dense_run_01/01_train/usi.csv"

    # with open(score_filename, 'w', newline='', buffering=1) as csvfile:
    #     csvwriter = csv.writer(csvfile, delimiter=',',
    #                             quotechar='|', quoting=csv.QUOTE_MINIMAL)
    #     for i in range(13, 26, 1):
    #         for j in range(0, 4, 1):
    #             try:
    #                 eval_file = f'/nrs/funke/wolfs2/lisl/experiments/dense_run_01/01_train/setup_t{i:04}/evaluation/rotflip/00047999/embedding_{j}_0.zarr'

    # score_filename = "/nrs/funke/wolfs2/lisl/experiments/dense_run_01/01_train/bbbc.csv"

    # with open(score_filename, 'w', newline='', buffering=1) as csvfile:
    #     csvwriter = csv.writer(csvfile, delimiter=',',
    #                             quotechar='|', quoting=csv.QUOTE_MINIMAL)
    #     for i in range(0, 13, 1):
    #         for j in range(0, 10, 1):


                # try:
                #     eval_file = f'/nrs/funke/wolfs2/lisl/experiments/dense_run_01/01_train/setup_t{i:04}/evaluation/rotflip/00025999/embedding_{j}_0.zarr'

                asv = AnchorSegmentationValidation()

                z_array = zarr.open(eval_file, mode="r+")
                print(list(z_array.keys()))
                emb = np.array(z_array["0/embedding_abs"])[None]
                fg = (np.array(z_array["0/threshold_li"]) > 0)

                x = torch.from_numpy(np.array(z_array["0/raw"]))[None]

                seg_results = asv.mws_segmentation(torch.from_numpy(emb), np.logspace(0, 4, base=10., num=25), foreground_mask=fg)#, img_dir_and_prefix=("/groups/funke/home/wolfs2/tmp/mws", "xxx"), output_z_array=None)
                # gtseg = (np.array(z_array["0/gt_segmentation"]) > 0)
                folder = f"/groups/funke/home/wolfs2/tmp/dsb_later/{i}/{j}"
                os.makedirs(folder, exist_ok=True)
                # asv.visualizalize_segmentation_dict(seg_results, x, f"{folder}/mws_")

                seg_scores = []
                for k in seg_results:
                    for jj, seg in enumerate(seg_results[k]):
                        gt_segmentation = (np.array(z_array[f"{jj}/gt_segmentation"])).astype(np.int64)
                        # print(k, seg_metric(seg, gt_segmentation[j]))
                        sc=matching_dataset([gt_segmentation[0]], [seg], thresh=0.5, show_progress = False)
                        seg_scores.append(sc.accuracy)
                        print(k, j, jj, sc.accuracy)
                        csvwriter.writerow([k, i, j, jj, sc.accuracy])

                # csvwriter.writerow([i, j, np.array(seg_scores).max()])




    # emb_shape = embedding.shape
    # img_shape = embedding.shape[-len(offsets[0]):]
    # n_img_dims = len(img_shape)

    # if repulsive_strides is None:
    #     repulsive_strides = (1,) * (n_img_dims - 2) + (8, 8)
    # repulsive_strides = np.array(repulsive_strides, dtype=int)

    # # actually not needed, huge offsets are fine
    # # for off in offsets:
    # #    assert all(abs(o) < s for o, s in zip(off, emb.shape[-len(off):])), \
    # #        f'offset {off} is to big for image of shape {img_shape}'
    # with torch.no_grad():
    #     embedding = embedding.cpu()
    #     if affinity_measure is not None:
    #         affinities = embedding_to_affinities(embedding, offsets=offsets, affinity_measure=affinity_measure,
    #                                              pass_offset=pass_offset)
    #     else:
    #         affinities = embedding
    #     affinities = affinities.contiguous().view((-1, len(offsets)) + emb_shape[-n_img_dims:])

    #     if percentile is not None:
    #         affinities -= np.percentile(affinities, percentile)
    #         affinities[:, :ATT_C] *= -1
    #     else:
    #         affinities[:, :ATT_C] *= -1
    #         affinities[:, :ATT_C] += 1
    #     affinities[:, :ATT_C] *= attraction_factor
    #     if z_delay != 0:
    #         affinities[:, (offsets[:, 0] != 0).astype(np.uint8)] += z_delay

    #     result = []
    #     for aff in affinities:
    #         seg = compute_mws_segmentation(aff,
    #                                        offsets,
    #                                        ATT_C,
    #                                        strides=repulsive_strides,
    #                                        mask=mask).astype(np.int32)
    #         result.append(seg)

    #     result = np.stack(result, axis=-(n_img_dims + 1)).reshape(emb_shape[:-n_img_dims - 1] + emb_shape[-n_img_dims:])

    #     if return_affinities:
    #         return torch.from_numpy(result), affinities
    #     else:
    #         return torch.from_numpy(result)


