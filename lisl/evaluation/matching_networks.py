import os
import torch
from tqdm import tqdm
import logging
import numpy as np
from torchmeta.datasets.fastdataset import FastCombinationMetaDataset
from torchmeta.datasets.helpers import fast_dataset_creator, omniglot

from torchmeta.utils.data import BatchMetaDataLoader
from torchmeta.utils.matching import matching_log_probas, matching_loss, spatial_matching_loss, spatial_matching_log_probas

import torch.nn as nn
from torch.utils.data.dataset import Dataset
import sklearn.metrics

from lisl.utils.sampling import roundrobin_break_early

from lisl.models.model import MLP


class MatchingNetwork(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_size=512):
        super(MatchingNetwork, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_size = hidden_size

        self.ndim = 2

        self.spatial_encoder = MLP(in_channels,
                                   self.ndim,
                                   n_hidden=hidden_size,
                                   n_hidden_layers=4,
                                   p=0.1,
                                   ndim=0)
        
        self.semantic_encoder = MLP(in_channels,
                                    out_channels,
                                    n_hidden=hidden_size,
                                    n_hidden_layers=4,
                                    p=0.1,
                                    ndim=0)

    def forward(self, inputs):
        
        abs_coords = inputs[..., :self.ndim]
        abs_coords = abs_coords.reshape(-1, self.ndim)
        z = inputs[..., self.ndim:]
        z = z.reshape(-1, *z.shape[2:])
        print("z.shape", z.shape)
        semantic_embeddings = self.semantic_encoder(z)
        semantic_embeddings = semantic_embeddings.view(*inputs.shape[:2], -1)

        spatial_embeddings = self.spatial_encoder(z)
        spatial_embeddings += abs_coords
        spatial_embeddings = spatial_embeddings.view(*inputs.shape[:2], -1)
        
        return semantic_embeddings, spatial_embeddings


logger = logging.getLogger(__name__)


def train(args):
    logger.warning('This script is an example to showcase the extensions and '
                   'data-loading features of Torchmeta, and as such has been '
                   'very lightly tested.')
    datasets = []
    for root_idx in range(0, 100):
        folders = {
            "raw": ("/nrs/funke/wolfs2/lisl/datasets/dsb_indexed.zarr", f"train/raw/{root_idx}"),
            "gt_segmentation": ("/nrs/funke/wolfs2/lisl/datasets/dsb_indexed.zarr", f"train/gt_segmentation/{root_idx}"),
            "embedding": ("/nrs/funke/wolfs2/lisl/experiments/semantic/c32/prediction/anchor.zarr", f"train/prediction_interm/{root_idx}"),
            "min_samples": 10}
        
        ds = fast_dataset_creator(folders,
                        shots=args.num_shots,
                        ways=args.num_ways,
                        shuffle=True,
                        test_shots=3,
                        transform=None,
                        meta_train=True)

        print("ds ", root_idx, " loaded")

        datasets.append(ds)

    loaders = []
    for ds in datasets:
        loaders.append(BatchMetaDataLoader(ds,
                                        batch_size=args.batch_size,
                                        shuffle=True,
                                        num_workers=args.num_workers))

    model = MatchingNetwork(512,
                            args.embedding_size,
                            hidden_size=args.hidden_size)
    model.to(device=args.device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print("starting")

    accuracy_window = {}
    accuracy_window["precision"] = []
    accuracy_window["recall"] = []
    accuracy_window["fscore"] = []

    # Training loop
    for epoch in range(50):

        dataloader = roundrobin_break_early(*loaders)

        with tqdm(dataloader, total=args.num_batches) as pbar:
            for batch_idx, batch in enumerate(pbar):
                model.zero_grad()

                # import pdb
                # pdb.set_trace()
                # print(batch)

                train_inputs, train_targets = batch['train']
                train_inputs = train_inputs.to(device=args.device)
                train_targets = train_targets.to(device=args.device)
                train_semantic_embeddings, train_spatial_embeddings = model(train_inputs)

                test_inputs, test_targets = batch['test']
                test_inputs = test_inputs.to(device=args.device)
                test_targets = test_targets.to(device=args.device)
                test_semantic_embeddings, test_spatial_embeddings = model(test_inputs)

                loss = spatial_matching_loss(train_spatial_embeddings,
                                             train_semantic_embeddings,
                                             test_spatial_embeddings,
                                             test_semantic_embeddings,
                                             train_targets,
                                             test_targets,
                                             args.num_ways,
                                             eps=1e-8)

                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    # calculate the accuracy
                    log_probas = spatial_matching_log_probas(train_spatial_embeddings,
                                                             train_semantic_embeddings,
                                                             test_spatial_embeddings,
                                                             test_semantic_embeddings,
                                                             train_targets,
                                                             test_targets,
                                                             args.num_ways,
                                                             eps=1e-8)

                    test_predictions = torch.argmax(log_probas, dim=1)
                    precision, recall, fscore, _ = sklearn.metrics.precision_recall_fscore_support(test_targets.cpu().numpy().ravel(),
                                                                                                   test_predictions.cpu().numpy().ravel(),
                                                                                                   average="macro")
                    if len(accuracy_window["precision"]) > 100:
                        accuracy_window["precision"].pop(0)
                        accuracy_window["recall"].pop(0)
                        accuracy_window["fscore"].pop(0)

                    accuracy_window["precision"].append(precision)
                    accuracy_window["recall"].append(recall)
                    accuracy_window["fscore"].append(fscore)

                    pbar.set_postfix(precision='{0:.4f}'.format(np.mean(accuracy_window["precision"])),
                    recall='{0:.4f}'.format(np.mean(accuracy_window["recall"])),
                    fscore='{0:.4f}'.format(np.mean(accuracy_window["fscore"])))

                if pbar.n > args.num_batches:
                    break

        # Save model
        if args.output_folder is not None:
            filename = os.path.join(args.output_folder, 
                f'matching_network_{epoch}_{args.num_shots}shot_{args.num_ways}way_spatial_{fscore:.4f}.pytorch')
            print("saving", filename)
            torch.save(model.state_dict(), filename)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser('Matching Networks')

    parser.add_argument('folder', type=str,
        help='Path to the folder the data is downloaded to.')
    parser.add_argument('--output_folder', type=str,
        help='Path, where the model is saved to.')
    parser.add_argument('--num-shots', type=int, default=5,
        help='Number of examples per class (k in "k-shot", default: 5).')
    parser.add_argument('--num-ways', type=int, default=5,
        help='Number of classes per task (N in "N-way", default: 5).')

    parser.add_argument('--embedding-size', type=int, default=64,
        help='Dimension of the embedding/latent space (default: 64).')
    parser.add_argument('--hidden-size', type=int, default=64,
        help='Number of channels for each convolutional layer (default: 64).')

    parser.add_argument('--output-folder', type=str, default=None,
        help='Path to the output folder for saving the model (optional).')
    parser.add_argument('--batch-size', type=int, default=16,
        help='Number of tasks in a mini-batch of tasks (default: 16).')
    parser.add_argument('--num-batches', type=int, default=100,
        help='Number of batches the prototypical network is trained over (default: 100).')
    parser.add_argument('--num-workers', type=int, default=1,
        help='Number of workers for data loading (default: 1).')
    parser.add_argument('--download', action='store_true',
        help='Download the Omniglot dataset in the data folder.')
    parser.add_argument('--use-cuda', action='store_true',
        help='Use CUDA if available.')

    args = parser.parse_args()
    args.device = torch.device('cuda' if args.use_cuda
        and torch.cuda.is_available() else 'cpu')

    train(args)
