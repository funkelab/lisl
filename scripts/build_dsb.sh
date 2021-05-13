
#BSUB -o dsb_build10.log
#BSUB -n 1

# python build_fast_ds.py --out_array /nrs/funke/wolfs2/lisl/datasets/fast_dsb_fin.zarr --img_out_array /nrs/funke/wolfs2/lisl/datasets/fast_dsb_fin.zarr --gt_array /nrs/funke/wolfs2/lisl/datasets/dsb.zarr --gt_key train/gt_segmentation --emb_arrays /nrs/funke/wolfs2/lisl/experiments/semantic/c32/prediction/anchor.zarr /nrs/funke/wolfs2/lisl/experiments/semantic/c32/prediction/semantic.zarr /nrs/funke/wolfs2/lisl/datasets/dsb_indexed_backup.zarr --emb_keys train/prediction_interm train/prediction train/raw --bg_distance 16
#python build_fast_ds.py --out_array /nrs/funke/wolfs2/lisl/datasets/fast_dsb_test.zarr --img_out_array /nrs/funke/wolfs2/lisl/datasets/fast_dsb_img_test.zarr --gt_array /nrs/funke/wolfs2/lisl/datasets/dsb.zarr --gt_key test/gt_segmentation --emb_arrays /nrs/funke/wolfs2/lisl/experiments/semantic/c32/prediction_test/anchor_test.zarr /nrs/funke/wolfs2/lisl/experiments/semantic/c32/prediction_test/semantic_test.zarr /nrs/funke/wolfs2/lisl/datasets/dsb.zarr --emb_keys test/prediction_interm test/prediction test/raw --bg_distance 16

#python build_coord_ds.py
#  --out_array /nrs/funke/wolfs2/lisl/datasets/fast_dsb_coord.zarr
#  --gt_array /nrs/funke/wolfs2/lisl/datasets/dsb_indexed_backup.zarr
#  --gt_key train/gt_segmentation
#  --emb_arrays /nrs/funke/wolfs2/lisl/experiments/semantic/c32/prediction/anchor.zarr /nrs/funke/wolfs2/lisl/experiments/semantic/c32/prediction/semantic.zarr /nrs/funke/wolfs2/lisl/datasets/dsb_indexed_backup.zarr
#  --emb_keys train/prediction_interm train/prediction train/raw
#  --raw_array /nrs/funke/wolfs2/lisl/datasets/dsb_indexed_backup.zarr
#  --raw_key train/raw
#  --bg_distance 16
# python build_coord_ds.py
#  --out_array /nrs/funke/wolfs2/lisl/datasets/fast_dsb_coord_test.zarr
#  --gt_array /nrs/funke/wolfs2/lisl/datasets/dsb.zarr
#  --gt_key test/gt_segmentation
#  --emb_arrays /nrs/funke/wolfs2/lisl/experiments/semantic/c32/prediction_test/anchor_test.zarr /nrs/funke/wolfs2/lisl/experiments/semantic/c32/prediction_test/semantic_test.zarr /nrs/funke/wolfs2/lisl/datasets/dsb.zarr
#  --emb_keys test/prediction_interm test/prediction test/raw
#  --raw_array /nrs/funke/wolfs2/lisl/datasets/dsb.zarr
#  --raw_key test/raw
#  --bg_distance 16

#python build_coord_ds.py
#  --out_array /nrs/funke/wolfs2/lisl/datasets/fast_dsb_coord2.zarr
#  --gt_array /nrs/funke/wolfs2/lisl/datasets/dsb_indexed_backup.zarr
#  --gt_key train/gt_segmentation
#  --emb_arrays /nrs/funke/wolfs2/lisl/experiments/semantic/c32/prediction/anchor.zarr /nrs/funke/wolfs2/lisl/experiments/semantic/c32/prediction/anchor.zarr /nrs/funke/wolfs2/lisl/experiments/semantic/c32/prediction/semantic.zarr /nrs/funke/wolfs2/lisl/datasets/dsb_indexed_backup.zarr
#  --emb_keys train/prediction train/prediction_interm train/prediction train/raw
#  --raw_array /nrs/funke/wolfs2/lisl/datasets/dsb_indexed_backup.zarr
#  --raw_key train/raw
#  --bg_distance 16
# python build_coord_ds.py
#  --out_array /nrs/funke/wolfs2/lisl/datasets/fast_dsb_coord_test2.zarr
#  --gt_array /nrs/funke/wolfs2/lisl/datasets/dsb.zarr
#  --gt_key test/gt_segmentation
#  --emb_arrays /nrs/funke/wolfs2/lisl/experiments/semantic/c32/prediction_test/anchor_test.zarr /nrs/funke/wolfs2/lisl/experiments/semantic/c32/prediction_test/anchor_test.zarr /nrs/funke/wolfs2/lisl/experiments/semantic/c32/prediction_test/semantic_test.zarr /nrs/funke/wolfs2/lisl/datasets/dsb.zarr
#  --emb_keys test/prediction test/prediction_interm test/prediction test/raw
#  --raw_array /nrs/funke/wolfs2/lisl/datasets/dsb.zarr
#  --raw_key test/raw
#  --bg_distance 16

# /groups/funke/home/wolfs2/miniconda3/envs/pytorch/bin/python /groups/funke/home/wolfs2/local/src/lisl/scripts/build_coord_ds.py --out_array /nrs/funke/wolfs2/lisl/datasets/fast_dsb_coord2.zarr --gt_array /nrs/funke/wolfs2/lisl/datasets/dsb_indexed_backup.zarr  --gt_key train/gt_segmentation --emb_arrays /nrs/funke/wolfs2/lisl/experiments/semantic/c32/prediction/anchor.zarr /nrs/funke/wolfs2/lisl/experiments/semantic/c32/prediction/anchor.zarr /nrs/funke/wolfs2/lisl/datasets/dsb_indexed_backup.zarr --emb_keys train/prediction train/prediction_interm train/raw  --raw_array /nrs/funke/wolfs2/lisl/datasets/dsb_indexed_backup.zarr --raw_key train/raw --bg_distance 16
/groups/funke/home/wolfs2/miniconda3/envs/pytorch/bin/python /groups/funke/home/wolfs2/local/src/lisl/scripts/build_coord_ds.py --out_array /nrs/funke/wolfs2/lisl/datasets/fast_dsb_coord_test2.zarr --gt_array /nrs/funke/wolfs2/lisl/datasets/dsb.zarr  --gt_key test/gt_segmentation --emb_arrays /nrs/funke/wolfs2/lisl/experiments/semantic/c32/prediction_test/anchor_test.zarr /nrs/funke/wolfs2/lisl/experiments/semantic/c32/prediction_test/anchor_test.zarr /nrs/funke/wolfs2/lisl/datasets/dsb.zarr --emb_keys test/prediction test/prediction_interm test/raw  --raw_array /nrs/funke/wolfs2/lisl/datasets/dsb.zarr --raw_key test/raw --bg_distance 16
