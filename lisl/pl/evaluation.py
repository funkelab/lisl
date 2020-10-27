from lisl.predict.volume import predict_volume
from pytorch_lightning.callbacks import Callback
import argparse
import inspect
import torch
import os
import copy
from lisl.datasets import Dataset
import zarr

# # from research_sslcell.imagetransforms import CPCTestTransformsCTC, CPCTestLabelTransformsCTC
# from torch.nn import functional as F
# from torch import nn
# from PIL import Image

# import pytorch_lightning as pl
# import torch
# import torch.optim as optim


# from pytorch_lightning.utilities import rank_zero_warn

# from pl_bolts.callbacks.self_supervised import SSLOnlineEvaluator
# import pl_bolts

# import matplotlib

# from skimage.io import imsave
# import numpy as np
# import stardist
# from sklearn.decomposition import PCA
# from research_sslcell.utils import UnPatchify, adapted_rand, tozeroone
# from skimage import measure
# from skimage.morphology import watershed
# from scipy import ndimage
# import math
# from tqdm import tqdm

# from pl_bolts.losses.self_supervised_learning import CPCTask

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

    # def log_img_and_seg(self, img, label, predictions, prefix, pl_module, predicted_labels=None, score=None, embeddings=None):

    #     def label2color(label):

    #         if isinstance(label, Image.Image):
    #             label = np.array(label)
    #             if len(label.shape) == 3:
    #                 label = label[..., 0]

    #         cmap = matplotlib.cm.get_cmap('nipy_spectral')
    #         shuffle_labels = np.concatenate(
    #             ([0], np.random.permutation(label.max()) + 1))
    #         label = shuffle_labels[label]
    #         return cmap(label / label.max()).transpose(2, 0, 1)

    #     def vis(x):
    #         if isinstance(x, Image.Image):
    #             x = np.array(x)

    #         assert(len(x.shape) in [2, 3])

    #         if len(x.shape) == 2:
    #             x = x[None]
    #         else:
    #             if x.shape[0] not in [1, 3]:
    #                 if x.shape[2] in [1, 3]:
    #                     x = x.transpose(2, 0, 1)
    #                 else:
    #                     raise Exception(
    #                         "can not visualize array with shape ", x.shape)

    #         return x

    #     pl_module.logger.experiment.add_image(f'{prefix}_img', vis(img), pl_module.global_step)
    #     pl_module.logger.experiment.add_image(f'{prefix}_gt', label2color(label), pl_module.global_step)

    #     _, max_class = torch.max(predictions, 1)
    #     three_class_probs = torch.nn.functional.softmax(
    #         predictions, dim=1)

    #     pl_module.logger.experiment.add_image(f'{prefix}_3class',
    #                                           torch.stack(
    #                                               (max_class[0] == 0, max_class[0] == 1, max_class[0] == 2), dim=0),
    #                                           pl_module.global_step)
    #     pl_module.logger.experiment.add_image(f'{prefix}_probs', three_class_probs[0], pl_module.global_step)

    #     if predicted_labels is not None:
    #         pl_module.logger.experiment.add_image(f'{prefix}_instances', label2color(predicted_labels), pl_module.global_step)

    #     if score is not None:
    #         pl_module.logger.log_metrics(score, step=pl_module.global_step)

    #     if embeddings is not None:
    #         _, C, W, H = embeddings.shape
    #         pca_in = embeddings[0].detach().cpu().numpy().reshape(C, -1).T
    #         pca = PCA(n_components=3, whiten=True)
    #         pca_out = pca.fit_transform(pca_in).T
    #         pca_image = pca_out.reshape(3, W, H)
    #         pl_module.logger.experiment.add_image(f'{prefix}_PCA', pca_image, pl_module.global_step)
    #         pl_module.logger.experiment.add_image(f'{prefix}_embedding_0_2', embeddings[0, :3], pl_module.global_step)

    def compute_3class_labels(self, train_label, patch_size, device):

        train_seg_gt = np.array(train_label)[..., 0]
        train_stardist = stardist.geometry.star_dist(train_seg_gt, n_rays=32)

        background = train_seg_gt == 0
        inner = train_stardist.min(axis=-1) > 3
        # classes 0: boundaries, 1: inner_cell, 2: background
        threeclass = (2 * background) + inner

        # remove pixels that are not the center of any patch
        lc = (patch_size // 2)
        rc = patch_size - lc - 1

        train_seg_gt = train_seg_gt[..., lc:-rc, lc:-rc]
        train_threeclass = threeclass[..., lc:-rc, lc:-rc]
        train_threeclass = torch.from_numpy(
            train_threeclass[None]).to(device)

        return train_threeclass, train_seg_gt

    def compute_3class_segmentation(self, predictions):

        _, max_3class = torch.max(predictions, 1)

        predicted_seg = []
        inner = max_3class.cpu().numpy() == 1
        background = max_3class.cpu().numpy() == 2

        for batch in range(max_3class.shape[0]):

            seeds = measure.label(inner[batch], background=0)
            distance = ndimage.distance_transform_edt(1 - background[batch])
            predicted_seg.append(
                watershed(-distance, seeds, mask=1 - background[batch]))

        return np.stack(predicted_seg)

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


        z_array = zarr.open(os.path.join((eval_directory, embeddings.zarr)))

        import pdb
        pdb.set_trace()

        print("done")

        # Todo: Train MLP for (star dist, 3 class, affinity + MWS)

        # Todod: Measure Segmentation performance

        return


        print("segmentation inference")

        train_img, train_label, test_img, test_label = self.load_train()

        tf = CPCTestTransformsCTC(patch_size=trainer.datamodule.patch_size,
                                  overlap=trainer.datamodule.patch_size - 1,
                                  image_resize_factor=trainer.datamodule.image_resize_factor,
                                  patchify=True)

        ltf = CPCTestLabelTransformsCTC(patch_size=trainer.datamodule.patch_size,
                                        overlap=trainer.datamodule.patch_size - 1,
                                        image_resize_factor=trainer.datamodule.image_resize_factor)

        train_label = ltf(train_label)
        test_label = ltf(test_label)

        train_threeclass, _ = self.compute_3class_labels(
            train_label, trainer.datamodule.patch_size, pl_module.device)
        test_threeclass, gt_seg = self.compute_3class_labels(
            test_label, trainer.datamodule.patch_size, pl_module.device)

        with torch.no_grad():
            patches = tf(train_img)
            embeddings = pl_module.forward_memreduced(patches[None]).detach()

        model, train_predictions = self.train_simple_model(
            embeddings, train_threeclass, pl_module)
        train_predicted_seg = self.compute_3class_segmentation(
            train_predictions)[0]

        self.log_img_and_seg(np.array(ltf(train_img)).transpose(2, 0, 1),
                             train_label,
                             train_predictions,
                             "train",
                             pl_module,
                             predicted_labels=train_predicted_seg,
                             embeddings=embeddings)

        with torch.no_grad():
            test_patches = tf(test_img)

            # test_embeddings = pl_module(test_patches.cuda()[None])
            test_embeddings, preds, targets = pl_module.forward_memreduced(
                test_patches[None], return_prediction_and_target=True, )

            # this is the prediction offset
            for k, i in enumerate(range(pl_module.contrastive_task.steps_to_ignore,
                                        pl_module.contrastive_task.steps_to_predict)):
                c = targets.shape[1]
                dist = i * (trainer.datamodule.patch_size -
                            trainer.datamodule.patch_overlap) + 1
                shifted_preds = preds[:, k*c:(k+1)*c, :-dist, :]
                inv_shifted_target = targets[:, :, :-dist, :]
                shifted_targets = targets[:, :, dist:, :]

                sim1 = 0.5 + 0.5 * \
                    cosine_similarity(
                        shifted_preds, shifted_targets, dim=1, eps=1e-8)

                pl_module.logger.experiment.add_image(
                    f'cosine_similarity_d{i}', tozeroone(sim1[0, None]), pl_module.global_step)

                sim2 = 0.5 + 0.5 * \
                    cosine_similarity(inv_shifted_target,
                                      shifted_targets, dim=1, eps=1e-8)

                pl_module.logger.experiment.add_image(
                    f'cosine_similarity_with_shifted_target_d{i}', tozeroone(sim2[0, None]), pl_module.global_step)

            test_predictions = model(test_embeddings)
            test_predicted_seg = self.compute_3class_segmentation(
                test_predictions)[0]

            arand = adapted_rand(test_predicted_seg, gt_seg,
                                 ignore_label=True, all_stats=True)

            self.log_img_and_seg(test_img,
                                 test_label,
                                 test_predictions,
                                 "val",
                                 pl_module,
                                 predicted_labels=test_predicted_seg,
                                 score=arand,
                                 embeddings=test_embeddings)
