import numpy as np
from scipy.ndimage.segmentation import distance_transform_edt
import zarr
import argparse
from tqdm import tqdm

def duplicate_and_interpolate(embeddings):
    mixed = embeddings[np.random.permutation(len(embeddings))]
    return np.concatenate((embeddings,
    (embeddings + mixed) / 2),
    axis=0)

def convert_ds(out_array, img_out_array, gt_array, gt_key, emb_arrays, emb_keys, min_samples=30, bg_distance=40):

    ds_zarr = zarr.open(gt_array, "r")
    out_array = zarr.open(out_array, "a")
    img_out_array = zarr.open(img_out_array, "a")
    
    for img_idx in tqdm(ds_zarr[gt_key]):

        if img_idx in out_array:
            print(f"skipping frame {img_idx}", )
            continue
        else:
            out_array.create_group(img_idx)

        gt_key_img_key = f"{gt_key}/{img_idx}"
        gt_segmentation = ds_zarr[gt_key_img_key][:]

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

        inst_embedding = np.concatenate(inst_embedding, axis=0)

        x = np.arange(inst_embedding.shape[-1], dtype=np.float32)
        y = np.arange(inst_embedding.shape[-2], dtype=np.float32)

        coords = np.meshgrid(x, y, copy=True)
        print(" shapes ", coords[0].shape)
        print(" shapes ", coords[1].shape)
        print(" shapes ", inst_embedding.shape)
        print(
            f"gt_segmentation {gt_segmentation.shape}, inst_embedding {inst_embedding.shape}")
        inst_embedding = np.concatenate([coords[0:1],
                                         coords[1:2],
                                         inst_embedding], axis=0)

        instance_idx = 0
        bg_mask = gt_segmentation == 0
        for idx in tqdm(np.unique(gt_segmentation), leave=False):
            if idx == 0:
                img_out_array.create_dataset(f"emb/{img_idx}", data=inst_embedding, overwrite=True)
                img_out_array.create_dataset(f"gt/{img_idx}", data=gt_segmentation, overwrite=True)
                continue

            mask = gt_segmentation == idx
            instance_embedding = np.transpose(inst_embedding[:, mask])
            instance_embedding = instance_embedding.astype(np.float32)
            
            if len(instance_embedding) <= 1:
                continue

            while len(instance_embedding) < min_samples:
                instance_embedding = duplicate_and_interpolate(instance_embedding)

            assert len(instance_embedding) >= min_samples
            instance_embedding = instance_embedding[np.random.permutation(len(instance_embedding))]
            out_array.create_dataset(f"{img_idx}/{instance_idx}/foreground",
                                     data=instance_embedding,
                                     compressor=None,
                                     chunks=(16, instance_embedding.shape[1]),
                                     overwrite=True)

            bg_mask = gt_segmentation == 0
            # add a background samples in close proximity to the object
            background_distance = distance_transform_edt(
                gt_segmentation != idx)
            bg_close_to_instance_mask = np.logical_and(background_distance < bg_distance,
                                                    bg_mask)
            bg_far_from_instance_mask = np.logical_and(background_distance >= bg_distance,
                                                    bg_mask)

            bg_close_instance_embedding = np.transpose(
                inst_embedding[:, bg_close_to_instance_mask])
            bg_close_instance_embedding = bg_close_instance_embedding.astype(np.float32)
            # draw close background clicks equal to half the amount of foreground clicks
            bg_close_instance_embedding = bg_close_instance_embedding[np.random.permutation(len(bg_close_instance_embedding))[:len(instance_embedding)]]

            bg_far_from_instance_mask

            bg_far_from_instance_mask = np.transpose(
                inst_embedding[:, bg_far_from_instance_mask])
            bg_far_from_instance_mask = bg_far_from_instance_mask.astype(np.float32)
            # draw far background clicks equal to half the amount of foreground clicks
            bg_far_from_instance_mask = bg_far_from_instance_mask[np.random.permutation(len(bg_far_from_instance_mask))[:len(instance_embedding)]]

            # combine close and far clicks
            bg_instance_embedding = np.concatenate((bg_close_instance_embedding, bg_far_from_instance_mask))
            # shuffle close and far points randomly
            bg_instance_embedding = bg_instance_embedding[np.random.permutation(len(bg_instance_embedding))]
            
            assert len(bg_instance_embedding) >= min_samples
            out_array.create_dataset(f"{img_idx}/{instance_idx}/background",
                                     data=bg_instance_embedding,
                                     compressor=None,
                                     chunks=(16, instance_embedding.shape[1]),
                                     overwrite=True)
            instance_idx += 1


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--out_array", help="path to output dataset", required=True)
    parser.add_argument("--img_out_array", help="path to output image dataset", required=True)
    parser.add_argument("--gt_array", help="path to image dataset file", required=True)
    parser.add_argument("--gt_key", help="key to raw images inside of ds array", required=True)
    parser.add_argument("--emb_arrays", help="file of embedding array. embeddings will be appended", nargs='+', required=True)
    parser.add_argument("--emb_keys", help="keys to embeddings inside of emb_arrays. embeddings will be appended", nargs='+', required=True)
    parser.add_argument("--min_samples", default=50, type=int)
    parser.add_argument("--bg_distance", default=16, type=int)
    args = parser.parse_args()
    convert_ds(args.out_array,
               args.img_out_array,
               args.gt_array,
               args.gt_key,
               args.emb_arrays,
               args.emb_keys,
               min_samples=args.min_samples,
               bg_distance=args.bg_distance)
