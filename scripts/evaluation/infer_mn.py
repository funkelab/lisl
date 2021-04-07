import zarr
import torch.nn as nn
import torch
import numpy as np
from lisl.models.model import MLP
from sklearn.decomposition import PCA

from lisl.evaluation.matching_networks import MatchingNetwork

# load model
model_path = "/home/swolf/local/data/lisl/inferenced/00/matching_network_2_5shot_3way.pytorch"
embedding_file = "/home/swolf/local/data/lisl/inferenced/00/anchor.zarr"
embedding_key = "train/prediction_interm/300"

output_file = "/nrs/funke/wolfs2/lisl/experiments/semantic/c32/prediction/matching_network.zarr"
zarray_out = zarr.open(output_file, "w")

for frame in range(400):
    model_path = "/groups/funke/home/wolfs2/tmp/matchingnetworks/matching_network_33_5shot_5way_sem_spatial_0.9833.pytorch"
    model_path = "/groups/funke/home/wolfs2/tmp/matchingnetworks/matching_network_0_5shot_5way_sem_spatial_0.9834.pytorch"
    embedding_file = ("/nrs/funke/wolfs2/lisl/experiments/semantic/c32/prediction/anchor.zarr",
                    "/nrs/funke/wolfs2/lisl/experiments/semantic/c32/prediction/semantic.zarr")

    embedding_keys = (f"train/prediction_interm/{frame}", f"train/prediction/{frame}")

    model = MatchingNetwork(544, 64, hidden_size=64)
    model.load_state_dict(torch.load(model_path))
    model = model.eval()

    # load input embeddings
    emb = []
    for ef, ek in zip(embedding_file, embedding_keys):
        zarray = zarr.open(ef, "r+")
        emb.append(zarray[ek][:])
    print([x.shape for x in emb])
    emb = np.concatenate(emb, axis=1)


    x = np.arange(emb.shape[-1], dtype=np.float32)
    y = np.arange(emb.shape[-2], dtype=np.float32)

    with torch.no_grad():
        coords = np.meshgrid(x, y, copy=True)
        emb = np.concatenate([coords[0][None, None],
                            coords[1][None, None],
                            emb], axis=1)

        emb = torch.from_numpy(emb.astype(np.float32))

        emb = emb.permute(0,2,3,1).reshape(-1, 546)

        sem_prediction, spatial_prediction = model.forward(emb[None])

        spatial_prediction = spatial_prediction[0]
        sem_prediction = sem_prediction[0]
        
        pca_in = sem_prediction.cpu().numpy()

        pca = PCA(n_components=3, whiten=True)
        pca_out = pca.fit_transform(pca_in)

        pca_out = pca_out.reshape(256,256,3).transpose(2,0,1)
        zarray_out.create_dataset(f'train/{frame}/matching_network_prediciton_pca', data=pca_out, overwrite=True)
        spatial_prediction = spatial_prediction.view(256,256,-1).permute(2,0,1)
        sp_embedding = spatial_prediction.cpu().numpy()
        zarray_out.create_dataset(f'train/{frame}/matching_network_prediciton_spatial_absolute', data=sp_embedding, overwrite=True)
        abs_coords = np.concatenate([coords[0][None],
                                     coords[1][None]], axis=0)
        print(sp_embedding.shape, abs_coords.shape)
        sp_embedding -= abs_coords
        zarray_out.create_dataset(f'train/{frame}/matching_network_prediciton_spatial_relative', data=spatial_prediction.cpu().numpy(), overwrite=True)

        sem_prediction = sem_prediction.view(256,256,-1).permute(2,0,1)
        zarray_out.create_dataset(f'train/{frame}/matching_network_prediciton_semantic', data=sem_prediction.cpu().numpy(), overwrite=True)

        print("done with ", frame)
