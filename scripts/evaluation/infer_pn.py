import zarr
import torch.nn as nn
import torch
import numpy as np
from lisl.models.model import MLP
from sklearn.decomposition import PCA
from sklearn.cluster import MeanShift

from lisl.models.prototypical_network import PrototypicalNetwork

raw_gt_file = "/nrs/funke/wolfs2/lisl/datasets/dsb_indexed.zarr"
raw_gt_in = zarr.open(raw_gt_file, "r")

output_file = "/nrs/funke/wolfs2/lisl/experiments/semantic/c32/prediction/prototypical_network_04_12.zarr"
zarray_out = zarr.open(output_file, "w")

for frame in range(30):
    model_path = "/nrs/funke/wolfs2/lisl/experiments/fast/prototypical_networks_run_09/best_model.pth"
    embedding_file = ("/nrs/funke/wolfs2/lisl/experiments/semantic/c32/prediction/anchor.zarr",
                      "/nrs/funke/wolfs2/lisl/experiments/semantic/c32/prediction/semantic.zarr",
                      "/nrs/funke/wolfs2/lisl/datasets/dsb_indexed.zarr")
    max_dist = 3
    embedding_keys = (f"train/prediction_interm/{frame}", f"train/prediction/{frame}", f"train/raw/{frame}")

    edim = 512 + 32 + 1
    sdim = 2

    model = PrototypicalNetwork(edim, 4, 2)
    model.load_state_dict(torch.load(model_path))
    model = model.eval()

    # load input embeddings
    emb = []
    for ef, ek in zip(embedding_file, embedding_keys):
        zarray = zarr.open(ef, "r+")
        if zarray[ek].ndim == 2:
            emb.append(zarray[ek][:][None])
        elif zarray[ek].ndim == 3:
            emb.append(zarray[ek][:])
        elif zarray[ek].ndim == 4:
            emb.append(zarray[ek][0])
        else:
            raise NotImplementedError()

    emb = np.concatenate(emb, axis=0)
    x = np.arange(emb.shape[-1], dtype=np.float32)
    y = np.arange(emb.shape[-2], dtype=np.float32)
    width = emb.shape[-2]
    height = emb.shape[-1]

    with torch.no_grad():
        coords = np.meshgrid(x, y, copy=True)
        emb = np.concatenate([coords[0][None],
                              coords[1][None],
                              emb], axis=0)

        emb = torch.from_numpy(emb.astype(np.float32))
        emb = emb.permute(1, 2, 0).reshape(-1, edim+sdim)

        spatial_prediction, sem_prediction = model.forward(emb)

        bandwidth = 10
        ms = MeanShift(bandwidth=bandwidth)
        fgmask = sem_prediction.argmax(axis=1) == 1

        fg_pred = spatial_prediction[fgmask]

        print("ms fit start", fg_pred.shape)
        ms_seg = ms.fit_predict(fg_pred)
        print("ms pred end")
        print(ms_seg.shape)
        seg_img = np.zeros((width, height))
        seg_img[fgmask.reshape(width, height)] = ms_seg

        # ms_seg = ms_seg.reshape(width, height)
        zarray_out.create_dataset(f'train/{frame}/ms_seg', data=seg_img, overwrite=True)
        
        # pca_out = pca_out.reshape(width, height,  3).transpose(2, 0, 1)
        # zarray_out.create_dataset(f'train/{frame}/pn_prediciton_pca', data=pca_out, overwrite=True)
        spatial_prediction = spatial_prediction.view(width, height, -1).permute(2,0,1)
        sp_embedding = spatial_prediction.cpu().numpy()

        zarray_out.create_dataset(f'train/{frame}/pn_prediciton_spatial_absolute', data=sp_embedding, overwrite=True)
        abs_coords = np.concatenate([coords[0][None],
                                     coords[1][None]], axis=0)

        sp_embedding[:2] -= abs_coords
        zarray_out.create_dataset(f'train/{frame}/pn_prediciton_spatial_relative', data=spatial_prediction.cpu().numpy(), overwrite=True)
        dist = torch.from_numpy(sp_embedding[:2]).norm(2, dim=0).cpu().numpy()
        zarray_out.create_dataset(
            f'train/{frame}/pn_prediciton_dist', data=dist, overwrite=True)

        sem_prediction = sem_prediction.view(
            width, height, -1).permute(2, 0, 1)
        zarray_out.create_dataset(f'train/{frame}/pn_prediciton_semantic', data=sem_prediction.cpu().numpy(), overwrite=True)

        sem_class = sem_prediction.argmax(axis=0)
        zarray_out.create_dataset(
            f'train/{frame}/pn_prediciton_semantic_class', data=sem_class.cpu().numpy(), overwrite=True)

        zarray_out.create_dataset(
            f'train/{frame}/pn_seeds', data=sem_class.cpu().numpy() * (dist < max_dist), overwrite=True)

        gt_segmentation = raw_gt_in[f"train/gt_segmentation/{frame}"][:]
        zarray_out.create_dataset(
            f'train/{frame}/gt', data=gt_segmentation, overwrite=True)


        print("done with ", frame)
