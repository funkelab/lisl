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
from lisl.pl.utils import adapted_rand, vis, label2color, try_remove
from ctcmetrics.seg import seg_metric
from sklearn.decomposition import PCA

from PIL import Image
import matplotlib


def compute_3class_segmentation(inner, background):
    seeds = measure.label(inner, background=0)
    distance = ndimage.distance_transform_edt(1 - background)
    return watershed(-distance, seeds, mask=1 - background)


class SupervisedLinearSegmentationValidation(Callback):

    def __init__(self, test_ds_filename, test_ds_name_raw, test_ds_name_seg):
        self.test_filename = test_ds_filename
        self.test_ds_name_raw = test_ds_name_raw
        self.test_ds_name_seg = test_ds_name_seg
        super().__init__()

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=False)
        parser.add_argument('--test_ds_filename', type=str)
        parser.add_argument('--test_ds_name_raw', type=str)
        parser.add_argument('--test_ds_name_seg', type=str)
        return parser

    @classmethod
    def from_argparse_args(cls, args, **kwargs):
        """
        Adapted from the pytorch lightning trainer. Create an instance from CLI arguments. 
        Args:
            args: The parser or namespace to take arguments from. Only known arguments will be
                parsed and passed to the Callback.
            **kwargs: Additional keyword arguments that may override ones in the parser or namespace.
                These must be valid arguments.
        """
        if isinstance(args, argparse.ArgumentParser):
            args = cls.parse_argparser(args)
        params = vars(args)

        valid_kwargs = inspect.signature(cls.__init__).parameters
        callback_kwargs = dict((name, params[name])
                               for name in valid_kwargs if name in params)
        callback_kwargs.update(**kwargs)
        print(callback_kwargs)

        return cls(**callback_kwargs)

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
                imsave(os.path.join(eval_directory, name+".png"),
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
                           input_name='x',
                           checkpoint=None,
                           normalize_factor='skip',
                           model_output=0,
                           in_shape=(256, 256),
                           out_shape=(120, 120),
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
