import os
import os.path
import torch
from tqdm import tqdm
import logging
import numpy as np
from torchmeta.datasets.fastdataset import FastCombinationMetaDataset
from torchmeta.datasets.helpers import fast_dataset_creator, omniglot
import pickle
from torchmeta.utils.data import BatchMetaDataLoader
from torchmeta.utils.prototype import get_prototypes, prototypical_loss, get_accuracy

import torch.nn as nn
from torch.utils.data.dataset import Dataset
import sklearn.metrics

from lisl.utils.sampling import roundrobin_break_early
import time
from lisl.models.model import MLP
import torch.nn.functional as F

class PrototypicalNetwork(nn.Module):
    def __init__(self, in_channels, out_channels, n_sem_classes, hidden_size=512):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_size = hidden_size
        self.n_sem_classes = n_sem_classes
        self.ndim = 2
        self.spatial_instance_encoder = MLP(in_channels,
                                            out_channels,
                                   n_hidden=hidden_size,
                                   n_hidden_layers=4,
                                   p=0.1,
                                   ndim=0)
       
        self.semantic_encoder = MLP(in_channels,
                                    self.n_sem_classes,
                                    n_hidden=hidden_size,
                                    n_hidden_layers=4,
                                    p=0.1,
                                    ndim=0)


    def forward(self, inputs):
        
        abs_coords = inputs[..., :self.ndim]
        abs_coords = abs_coords.reshape(-1, self.ndim)
        z = inputs[..., self.ndim:]
        z = z.reshape(-1, *z.shape[2:])
        semantic_embeddings = self.semantic_encoder(z)
        semantic_embeddings = semantic_embeddings.view(*inputs.shape[:2], -1)

        spatial_instance_embeddings = self.spatial_instance_encoder(z)
        spatial_instance_embeddings[..., :self.ndim] += abs_coords
        spatial_instance_embeddings = spatial_instance_embeddings.view(*inputs.shape[:2], -1)
        
        return semantic_embeddings, spatial_instance_embeddings

logger = logging.getLogger(__name__)

def train(args):


    datasets = []
    for root_idx in tqdm(range(args.ds_size)):

        folders = {
            "index": root_idx,
            "cache": "/nrs/funke/wolfs2/lisl/datasets/prototypical_network_cache_uncompressed_5.zarr",
            "raw": ("/nrs/funke/wolfs2/lisl/datasets/dsb_indexed.zarr", f"train/raw/{root_idx}"),
            "gt_segmentation": ("/nrs/funke/wolfs2/lisl/datasets/dsb_indexed.zarr", f"train/gt_segmentation/{root_idx}"),
            "embedding": (("/nrs/funke/wolfs2/lisl/experiments/semantic/c32/prediction/anchor.zarr", f"train/prediction_interm/{root_idx}"),
                        ("/nrs/funke/wolfs2/lisl/experiments/semantic/c32/prediction/semantic.zarr", f"train/prediction/{root_idx}")),
            "min_samples": 10,
            "bg_distance": 20}
        
        ds = fast_dataset_creator(folders,
                                  shots=args.num_shots,
                                  ways=args.num_ways,
                                  shuffle=True,
                                  test_shots=3,
                                  transform=None,
                                  meta_train=True)

        if len(ds):
            datasets.append(ds)
        else:
            print(f"dataset with id {root_idx} appears to be empty. Will be skipped")

    os.makedirs(args.output_folder, exist_ok=True)

    loaders = []
    for ds in datasets:
        loaders.append(BatchMetaDataLoader(ds,
                                           batch_size=args.batch_size,
                                           shuffle=True,
                                           num_workers=args.num_workers))
        time.sleep(1)

    model = PrototypicalNetwork(544,
                            args.embedding_size,
                            2,
                            hidden_size=args.hidden_size)
    model.to(device=args.device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print("starting")

    accuracy_window = {}
    accuracy_window["inst_accuracy"] = []
    accuracy_window["sem_accuracy"] = []
    accuracy_window["combined"] = []

    # Training loop
    for epoch in range(50):

        dataloader = roundrobin_break_early(*loaders)

        with tqdm(dataloader, total=args.num_batches) as pbar:
            for batch_idx, batch in enumerate(pbar):
                model.zero_grad()

                train_inputs, train_instance_targets, train_semantic_targets = batch['train']
                train_inputs = train_inputs.to(device=args.device)
                train_instance_targets = train_instance_targets.to(device=args.device)
                train_semantic_targets = train_semantic_targets.to(device=args.device)

                train_semantic_embeddings, train_spatial_instance_embeddings = model(train_inputs)

                test_inputs, test_instance_targets, test_semantic_targets = batch['test']
                test_inputs = test_inputs.to(device=args.device)
                test_instance_targets = test_instance_targets.to(device=args.device)
                test_semantic_targets = test_semantic_targets.to(device=args.device)
                test_semantic_embeddings, test_spatial_instance_embeddings = model(test_inputs)

                # semantic loss
                c = train_semantic_embeddings.shape[-1]
                semantic_loss = F.cross_entropy(train_semantic_embeddings.view(-1, c),
                                                train_semantic_targets.view(-1))

                # semantic_prototypes = get_prototypes(train_semantic_embeddings, train_semantic_targets, 2)
                # semantic_loss = prototypical_loss(semantic_prototypes, test_semantic_embeddings, test_semantic_targets)

                # instance loss
                train_inst_emb = train_spatial_instance_embeddings
                test_inst_emb = test_spatial_instance_embeddings
                instance_prototypes = get_prototypes(
                    train_inst_emb, train_instance_targets, args.num_ways)
                instance_loss = prototypical_loss(instance_prototypes, test_inst_emb, test_instance_targets)

                loss = semantic_loss + instance_loss

                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    inst_accuracy = get_accuracy(instance_prototypes, test_inst_emb, test_instance_targets)
                    pred = test_semantic_embeddings.max(-1).indices
                    sem_accuracy = (pred == test_semantic_targets).sum().item() / (pred.size(0) * pred.size(1))
                    sem_accuracy = float(sem_accuracy)

                    if len(accuracy_window["inst_accuracy"]) > 100:
                        accuracy_window["inst_accuracy"].pop(0)
                        accuracy_window["sem_accuracy"].pop(0)
                        accuracy_window["combined"].pop(0)

                    accuracy_window["inst_accuracy"].append(float(inst_accuracy))
                    accuracy_window["sem_accuracy"].append(float(sem_accuracy))
                    accuracy_window["combined"].append(
                        float(inst_accuracy + sem_accuracy) / 2)

                    mean_inst_acc = np.mean(accuracy_window['inst_accuracy'])
                    mean_sem_acc = np.mean(accuracy_window['sem_accuracy'])
                    pbar.set_postfix(
                        inst_accuracy=f"{mean_inst_acc:.4f}",
                        sem_accuracy=f'{mean_sem_acc:.4f}',
                        loss=f"{loss:.4}")

                if pbar.n > args.num_batches:
                    break

        # Save model
        score = np.mean(accuracy_window["combined"])
        if args.output_folder is not None:
            filename = os.path.join(args.output_folder, 
                                    f'prototypical_networks_class_{epoch}_{args.num_shots}shot_{args.num_ways}way_{args.ds_size}_sem_spatial_{score:.4f}.pytorch')
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

    parser.add_argument('--ds_size', type=int, default=447,
        help='Number images in the dataset')

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
