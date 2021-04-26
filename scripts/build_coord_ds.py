import numpy as np
from scipy.ndimage.morphology import distance_transform_edt
import zarr
import argparse
from tqdm import tqdm

def duplicate(embeddings):
    return np.concatenate((embeddings, embeddings), axis=0)


def convert_ds(out_array, gt_array, gt_key, emb_arrays, emb_keys, raw_array, raw_key, min_samples=30, bg_distance=40):

    gt_ds_zarr = zarr.open(gt_array, "r")
    raw_ds_zarr = zarr.open(raw_array, "r")
    out_array = zarr.open(out_array, "a")
    
    for img_idx in tqdm(gt_ds_zarr[gt_key]):

        if img_idx in out_array:
            print(f"skipping frame {img_idx}", )
            continue
        else:
            out_array.create_group(img_idx)

        gt_key_img_key = f"{gt_key}/{img_idx}"
        gt_segmentation = gt_ds_zarr[gt_key_img_key][:]
        raw_img_data = raw_ds_zarr[f"{raw_key}/{img_idx}"][:]

        inst_embedding = []
        for emb_zarr_file, emb_key in zip(emb_arrays, emb_keys):
            emb_zarr = zarr.open(emb_zarr_file, "r")
            emb_key = emb_key
            if emb_zarr[emb_key][img_idx].ndim == 2:
                inst_embedding.append(emb_zarr[emb_key][img_idx][:][None])
            elif emb_zarr[emb_key][img_idx].ndim == 3:
                inst_embedding.append(emb_zarr[emb_key][img_idx][:])
            elif emb_zarr[emb_key][img_idx].ndim == 4:
                inst_embedding.append(emb_zarr[emb_key][img_idx][0])
            else:
                raise NotImplementedError()

        print([x.shape for x in inst_embedding])
        inst_embedding = np.concatenate(inst_embedding, axis=0)

        x = np.arange(inst_embedding.shape[-1], dtype=np.int32)
        y = np.arange(inst_embedding.shape[-2], dtype=np.int32)

        coords = np.meshgrid(x, y, copy=True)
        print(" shapes ", coords[0].shape)
        print(" shapes ", coords[1].shape)
        print(" shapes ", inst_embedding.shape)
        print(
            f"gt_segmentation {gt_segmentation.shape}, inst_embedding {inst_embedding.shape}")
        coords = np.concatenate([coords[0:1],
                                 coords[1:2]], axis=0).astype(np.int32)

        instance_idx = 0
        bg_mask = gt_segmentation == 0

        out_array.create_dataset(
            f"{img_idx}/gt", data=gt_segmentation.astype(np.int32), overwrite=True)
        out_array.create_dataset(
            f"{img_idx}/raw", data=raw_img_data.astype(np.float32), overwrite=True)
        out_array.create_dataset(f"{img_idx}/embedding", data=inst_embedding.astype(np.float32), overwrite=True)

        for idx in tqdm(np.unique(gt_segmentation), leave=False):
            if idx == 0:
                continue

            mask = gt_segmentation == idx
            # print(gt_key_img_key, f"{raw_key}/{img_idx}", "mask shape ",
            #       mask.shape, "coords.shape", coords.shape)
            instance_coords = np.transpose(coords[:, mask]).astype(np.int32)
            
            if len(instance_coords) <= 1:
                continue

            while len(instance_coords) < min_samples:
                instance_coords = duplicate(instance_coords)

            assert len(instance_coords) >= min_samples
            instance_coords = instance_coords[np.random.permutation(len(instance_coords))]
            out_array.create_dataset(f"{img_idx}/foreground/{instance_idx}",
                                     data=instance_coords,
                                     compressor=None,
                                     chunks=(16, instance_coords.shape[1]),
                                     overwrite=True)

            bg_mask = gt_segmentation == 0
            # add a background samples in close proximity to the object
            background_distance = distance_transform_edt(
                gt_segmentation != idx)
            bg_close_to_instance_mask = np.logical_and(background_distance < bg_distance,
                                                    bg_mask)
            bg_far_from_instance_mask = np.logical_and(background_distance >= bg_distance,
                                                    bg_mask)

            bg_close_instance_coords = np.transpose(coords[:, bg_close_to_instance_mask])
            bg_close_instance_coords = bg_close_instance_coords.astype(np.float32)
            # draw close background clicks equal to half the amount of foreground clicks
            bg_close_instance_coords = bg_close_instance_coords[np.random.permutation(len(bg_close_instance_coords))[:len(instance_coords)]]

            bg_far_from_instance_mask

            bg_far_from_instance_mask = np.transpose(coords[:, bg_far_from_instance_mask])
            bg_far_from_instance_mask = bg_far_from_instance_mask.astype(np.float32)
            # draw far background clicks equal to half the amount of foreground clicks
            bg_far_from_instance_coords = bg_far_from_instance_mask[np.random.permutation(len(bg_far_from_instance_mask))[:len(instance_coords)]]

            # combine close and far clicks
            bg_instance_coords = np.concatenate(
                (bg_close_instance_coords, bg_far_from_instance_coords))
            # shuffle close and far points randomly
            bg_instance_coords = bg_instance_coords[np.random.permutation(len(bg_instance_coords))]
            
            assert len(bg_instance_coords) >= min_samples
            out_array.create_dataset(f"{img_idx}/background/{instance_idx}",
                                     data=bg_instance_coords,
                                     compressor=None,
                                     chunks=(16, bg_instance_coords.shape[1]),
                                     overwrite=True)
            instance_idx += 1


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--out_array", help="path to output dataset", required=True)
    parser.add_argument("--gt_array", help="path to image dataset file", required=True)
    parser.add_argument("--gt_key", help="key to raw images inside of ds array", required=True)
    parser.add_argument(
        "--raw_array", help="path to image dataset file", required=True)
    parser.add_argument(
        "--raw_key", help="key to raw images inside of ds array", required=True)
    parser.add_argument("--emb_arrays", help="file of embedding array. embeddings will be appended", nargs='+', required=True)
    parser.add_argument("--emb_keys", help="keys to embeddings inside of emb_arrays. embeddings will be appended", nargs='+', required=True)
    parser.add_argument("--min_samples", default=50, type=int)
    parser.add_argument("--bg_distance", default=16, type=int)
    args = parser.parse_args()
    convert_ds(args.out_array,
               args.gt_array,
               args.gt_key,
               args.emb_arrays,
               args.emb_keys,
               args.raw_array,
               args.raw_key,
               min_samples=args.min_samples,
               bg_distance=args.bg_distance)
