from lisl.gp import (
    AddChannelDim,
    RemoveChannelDim,
    TransposeDims)
import daisy
import argparse
import gunpowder as gp
import logging
import lisl
from lisl.pl.model import PatchedResnet
from gunpowder.nodes.upsample import UpSample
from gunpowder.nodes.downsample import DownSample

logger = logging.getLogger(__name__)


def predict_frame(
        in_shape,
        out_shape,
        model_output,
        model_configfile,
        model_checkpoint,
        input_dataset_file,
        inference_frame,
        out_dir,
        out_filename,
        out_key_or_index=1,
        intermediate_layer=None,
        dataset_raw_key="train/raw",
        dataset_prediction_key="cooc_emb",
        dataset_intermediate_key="interm_cooc_emb",
        model_input_tensor_name="patches",
        model_architecture="PatchedResnet",
        num_workers=5,
        upsample=1):

    # initialize model
    if model_architecture == "PatchedResnet":
        model = PatchedResnet(1, 2, resnet_size=18)
    elif model_architecture == "unet":
        model = lisl.models.create(model_configfile)
    else:
        raise NotImplementedError(f"{model_architecture} not implemented")
    
    model.add_spatial_dim = True
    model.eval()

    # gp variables
    in_shape = gp.Coordinate(in_shape)
    out_shape = gp.Coordinate(out_shape)
    raw = gp.ArrayKey(f'RAW_{inference_frame}')
    raw_us = gp.ArrayKey(f'RAW_UPSAMPLED_{inference_frame}')
    prediction = gp.ArrayKey(f'PREDICTION_{inference_frame}')
    prediction_ds = gp.ArrayKey(f'PREDICTION_DS_{inference_frame}')
    intermediate_prediction = gp.ArrayKey(f'ITERM_{inference_frame}')

    ds_key = f'{inference_frame}/{dataset_raw_key}'
    out_key = f'{inference_frame}/{dataset_prediction_key}'
    interm_key = f'{inference_frame}/{dataset_intermediate_key}'

    # build pipeline
    zsource = gp.ZarrSource(input_dataset_file,
                            {raw: ds_key},
                            {raw: gp.ArraySpec(interpolatable=True, voxel_size=(upsample, upsample))})

    pipeline = zsource
    with gp.build(zsource):
        raw_roi = zsource.spec[raw].roi
        logger.info(f"raw_roi: {raw_roi}")

    pipeline += AddChannelDim(raw)
    pipeline += AddChannelDim(raw)

    pipeline += gp.Pad(raw, None)
    # setup prediction node
    pred_dict = {out_key_or_index: prediction}
    pred_spec = {prediction: gp.ArraySpec(roi=raw_roi)}
    if intermediate_layer is not None:
        pred_dict[intermediate_layer] = intermediate_prediction
        pred_spec[intermediate_prediction] = gp.ArraySpec(roi=raw_roi)

    if upsample > 1:
        pipeline += UpSample(raw, upsample, raw_us)
        netin = raw_us
    else:
        netin = raw

    pipeline += gp.torch.Predict(
        model,
        inputs={model_input_tensor_name: netin},
        outputs=pred_dict,
        array_specs=pred_spec,
        checkpoint=model_checkpoint,
        spawn_subprocess=True
    )

    if upsample > 1:
        pipeline += DownSample(prediction, upsample, prediction_ds)
    else:
        prediction_ds = prediction

    request = gp.BatchRequest()
    request.add(raw, in_shape)
    request.add(prediction_ds, out_shape)

    zarr_dict = {prediction_ds: out_key}
    if intermediate_layer is not None:
        zarr_dict[intermediate_prediction] = interm_key
        request.add(intermediate_prediction, out_shape)
    pipeline += gp.Scan(request, num_workers=num_workers)
    pipeline += gp.ZarrWrite(
        zarr_dict,
        output_dir=out_dir,
        output_filename=out_filename,
        compression_type='gzip')

    total_request = gp.BatchRequest()
    total_request[prediction_ds] = gp.ArraySpec(roi=raw_roi)
    if intermediate_layer is not None:
        total_request[intermediate_prediction] = gp.ArraySpec(roi=raw_roi)
    with gp.build(pipeline):
        pipeline.request_batch(total_request)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--in_shape', type=int, nargs='+')
    parser.add_argument('--out_shape', type=int, nargs='+')
    parser.add_argument('--model_output', default="semantic_embedding")
    parser.add_argument('--model_configfile', default=None)
    parser.add_argument('--model_checkpoint', default=None)
    parser.add_argument('--input_dataset_file')
    parser.add_argument('--out_dir')
    parser.add_argument('--out_filename')
    parser.add_argument('--inference_frame')
    parser.add_argument('--intermediate_layer', type=int, default=None)
    parser.add_argument('--upsample', type=int, default=1)
    parser.add_argument('--dataset_raw_key', default="train/raw")
    parser.add_argument('--dataset_prediction_key', default="train/prediction")
    parser.add_argument('--dataset_intermediate_key', default="interm_cooc_emb")
    
    parser.add_argument('--default_root_dir', default=None)
    parser.add_argument('--model_input_tensor_name', default="patches")
    parser.add_argument('--model_architecture', default="PatchedResnet")

    options = parser.parse_args()

    predict_frame(
        options.in_shape,
        options.out_shape,
        options.model_output,
        options.model_configfile,
        options.model_checkpoint,
        options.input_dataset_file,
        options.inference_frame,
        options.out_dir,
        options.out_filename,
        intermediate_layer=options.intermediate_layer,
        dataset_raw_key=options.dataset_raw_key,
        dataset_prediction_key=options.dataset_prediction_key,
        model_architecture=options.model_architecture,
        dataset_intermediate_key=options.dataset_intermediate_key,
        model_input_tensor_name=options.model_input_tensor_name,
        upsample=options.upsample)
