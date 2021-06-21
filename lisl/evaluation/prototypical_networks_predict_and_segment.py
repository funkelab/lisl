import argparse
import logging
import lisl
from lisl.models.prototypical_network import PrototypicalERFNetwork
import zarr
import torch
from tqdm import tqdm
import numpy as np
import time
logger = logging.getLogger(__name__)
from sklearn.cluster import MeanShift


def predict_and_segment(
        experiment_name,
        model_path,
        zarr_in_path,
        zarr_out_path,
        in_channels,
        inst_out_channels,
        n_sem_classes=2,
        ms_bandwidths=(3, 4, 5, 8, 12),
        gt_key="gt",
        emb_in_key="embedding",
        device="cuda"):

    out_array = zarr.open(zarr_out_path, "a")
    in_array = zarr.open(zarr_in_path, "a")

    with torch.no_grad():
        
        # load model
        model = PrototypicalERFNetwork(in_channels, inst_out_channels, n_sem_classes).to(device)
        model.load_state_dict(torch.load(model_path))

        emb_grp = f"inference/{experiment_name}/pn_embedding"
        if emb_grp not in out_array:
            out_array.create_group(emb_grp)
            out_array[emb_grp].attrs["model_path"] = model_path
            out_array[emb_grp].attrs["zarr_in_path"] = zarr_in_path
            out_array[emb_grp].attrs["in_channels"] = in_channels
            out_array[emb_grp].attrs["inst_out_channels"] = inst_out_channels


        for img_idx in tqdm(in_array):
            out_emb_key = f"{emb_grp}/{img_idx}"
            in_emb_key = f"{img_idx}/{emb_in_key}"
            # load input embedding
            if out_emb_key in out_array:
                print(f"skipping frame {img_idx}", )
                continue
            else:
                out_array.create_group(out_emb_key)

            if in_emb_key not in in_array:
                print(f"can not find {in_emb_key} in {zarr_in_path}") 
                print(f"keys {[k for k in in_array.keys()]}")
                continue

            input_embedding = in_array[in_emb_key][:]
            emb_channels, width, height = input_embedding.shape

            input_embedding = torch.from_numpy(input_embedding.astype(np.float32))
            spatial_instance_embeddings, semantic_prediction = model(input_embedding[None].to(device))

            semantic_prediction = semantic_prediction[0]
            spatial_instance_embeddings = spatial_instance_embeddings[0]

            # predict output embedding
            sem_key = f"{out_emb_key}/semantic_prediction"
            inst_key = f"{out_emb_key}/inst_embedding"

            out_array.create_dataset(sem_key,
                                     data=semantic_prediction.cpu().numpy(),
                                     overwrite=True)
            out_array.create_dataset(inst_key,
                                     data=spatial_instance_embeddings.cpu().numpy(),
                                     overwrite=True)

            for ms_bandwidth in ms_bandwidths:
                ms = MeanShift(bandwidth=ms_bandwidth)
                fgmask = semantic_prediction.cpu().numpy().argmax(axis=0) == 1
                fg_pred = spatial_instance_embeddings[:, fgmask].cpu().numpy().T

                start_time = time.time()
                print("start ms", fg_pred.shape, fgmask.sum() / fgmask.size)
                ms_seg = ms.fit_predict(fg_pred)

                print("prediction took ", time.time() - start_time)
                seg_img = np.zeros((width, height), dtype=np.int32)
                seg_img[fgmask.reshape(width, height)] = ms_seg + 1

                # ms_seg = ms_seg.reshape(width, height)
                ms_key = f"{out_emb_key}/ms_seg_bw{ms_bandwidth}"
                out_array.create_dataset(ms_key, data=seg_img, overwrite=True)
                out_array[ms_key].attrs["bandwidth"] = ms_bandwidth
                
                full_emb = spatial_instance_embeddings.cpu().view(spatial_instance_embeddings.shape[0], -1).numpy().T
                ms_seg_full = ms.predict(full_emb)
                ms_seg_full = ms_seg_full.astype(np.int32).reshape(width, height)

                # ms_seg = ms_seg.reshape(width, height)
                ms_key_full = f"{out_emb_key}/ms_seg_bw{ms_bandwidth}_full"
                out_array.create_dataset(ms_key_full, data=ms_seg_full, overwrite=True)
                out_array[ms_key_full].attrs["bandwidth"] = ms_bandwidth

                


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--experimentname", )
    parser.add_argument("--model_path", )
    parser.add_argument("--zarr_in_path", )
    parser.add_argument("--zarr_out_path", )
    parser.add_argument("--in_channels", type=int)
    parser.add_argument("--inst_out_channels", type=int)
    parser.add_argument("--n_sem_classes", default=2, type=int)
    parser.add_argument("--ms_bandwidths", default=(3, 4, 5, 8, 12), type=float, nargs='+')
    parser.add_argument("--gt_key", default="gt")
    parser.add_argument("--emb_in_key", default="embedding")
    parser.add_argument("--device", default="cuda")

    options = parser.parse_args()

    predict_and_segment(
        experiment_name=options.experimentname,
        model_path=options.model_path,
        zarr_in_path=options.zarr_in_path,
        zarr_out_path=options.zarr_out_path,
        in_channels=options.in_channels,
        inst_out_channels=options.inst_out_channels,
        n_sem_classes=options.n_sem_classes,
        ms_bandwidths=options.ms_bandwidths,
        gt_key=options.gt_key,
        emb_in_key=options.emb_in_key,
        device=options.device)

    # example command
    # python --experimentname dev00 --model_path /nrs/funke/wolfs2/lisl/experiments/pn_dsb_01/03_fast/setup_t0064/last_model.pth --zarr_in_path /nrs/funke/wolfs2/lisl/datasets/fast_dsb_img_test.zarr --zarr_out_path /nrs/funke/wolfs2/lisl/datasets/fast_dsb_img_test_out.zarr --in_channels 545 --inst_out_channels 4  --n_sem_classes 2
