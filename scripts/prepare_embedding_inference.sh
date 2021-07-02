# anchor_inference

python scripts/prepare_sls_inference.py -d /nrs/funke/wolfs2/lisl/experiments/ -e anchor_inference_interm_test -l ~/local/src/lisl/lisl/ -p /groups/funke/home/wolfs2/miniconda3/envs/pytorch/bin/python -r lisl/predict/predict_frame.py --args "--in_shape 79 79 --out_shape 64 64 --model_configfile config/semantic_train_embeddings.conf --model_checkpoint /nrs/funke/wolfs2/lisl/experiments/new_val_07/01_train/setup_t0045/models/model_00060599.torch --input_dataset_file /nrs/funke/wolfs2/lisl/datasets/dsb.zarr --out_dir /nrs/funke/wolfs2/lisl/experiments/semantic/c32/prediction_test --out_filename anchor_test.zarr --intermediate_layer 0 --dataset_raw_key test/raw" --max_frames=50

#python slurmexperimentmanager/train_all.py -d /nrs/funke/wolfs2/lisl/experiments/anchor_inference_interm_test/02_inference -s infer_*.sh --c 'bsub < ' 


# semantic inference
python scripts/prepare_sls_inference.py -d /nrs/funke/wolfs2/lisl/experiments/ -e semantic_interm_test -l ~/local/src/lisl/lisl/ -p /groups/funke/home/wolfs2/miniconda3/envs/pytorch/bin/python -r lisl/predict/predict_frame.py --args "--in_shape 180 180 --out_shape 88 88 --model_configfile /groups/funke/home/wolfs2/local/src/lisl/config/semantic_train_embeddings.conf --model_checkpoint /nrs/funke/wolfs2/lisl/experiments/semantic/c32/model_checkpoint_100000 --input_dataset_file /nrs/funke/wolfs2/lisl/datasets/dsb.zarr --out_dir /nrs/funke/wolfs2/lisl/experiments/semantic/c32/prediction_test --out_filename semantic_test.zarr --intermediate_layer 0 --dataset_raw_key test/raw --model_input_tensor_name raw_0 --model_architecture unet" --max_frames=50

# python3 slurmexperimentmanager/train_all.py -d /nrs/funke/wolfs2/lisl/experiments/semantic_interm_test/02_inference -s infer_*.sh --c 'bsub < '
