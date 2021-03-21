from lisl.gp import (
    AddChannelDim,
    RemoveChannelDim,
    TransposeDims)
import daisy
import argparse
import gunpowder as gp
import logging
import lisl

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
        dataset_prediction_key="train/prediction",
        dataset_intermediate_key="train/prediction_interm",
        num_workers=1):

    # initialize model
    model = lisl.models.create(model_configfile)
    model.eval()

    # gp variables
    in_shape = gp.Coordinate(in_shape)
    out_shape = gp.Coordinate(out_shape)
    input_name = "raw_0"
    raw = gp.ArrayKey(f'RAW_{inference_frame}')
    prediction = gp.ArrayKey(f'PREDICTION_{inference_frame}')
    intermediate_prediction = gp.ArrayKey(f'ITERM_{inference_frame}')

    ds_key = f'{dataset_raw_key}/{inference_frame}'
    out_key = f'{dataset_prediction_key}/{inference_frame}'
    interm_key = f'{dataset_intermediate_key}/{inference_frame}'

    # build pipeline
    zsource = gp.ZarrSource(input_dataset_file,
                            {raw: ds_key},
                            {raw: gp.ArraySpec(interpolatable=True, voxel_size=(1, 1))})

    pipeline = zsource

    context = (in_shape - out_shape) / 2
    with gp.build(zsource):
        raw_roi = zsource.spec[raw].roi
        logger.info(f"raw_roi: {raw_roi}")

    pipeline += AddChannelDim(raw)
    pipeline += AddChannelDim(raw)

    pipeline += gp.Pad(raw, context)
    # setup prediction node
    pred_dict = {out_key_or_index: prediction}
    pred_spec = {prediction: gp.ArraySpec(roi=raw_roi)}
    if intermediate_layer is not None:
        pred_dict[intermediate_layer] = intermediate_prediction
        pred_spec[intermediate_prediction] = gp.ArraySpec(roi=raw_roi)

    pipeline += gp.torch.Predict(
        model,
        inputs={input_name: raw},
        outputs=pred_dict,
        array_specs=pred_spec,
        checkpoint=model_checkpoint,
        spawn_subprocess=True
    )

    request = gp.BatchRequest()
    request.add(raw, in_shape)
    request.add(prediction, out_shape)

    zarr_dict = {prediction: out_key}
    if intermediate_layer is not None:
        zarr_dict[intermediate_prediction] = interm_key
        request.add(intermediate_prediction, out_shape)
    pipeline += gp.ZarrWrite(
        zarr_dict,
        output_dir=out_dir,
        output_filename=out_filename,
        compression_type='gzip') + \
        gp.Scan(request, num_workers=num_workers)

    total_request = gp.BatchRequest()
    with gp.build(pipeline):
        pipeline.request_batch(total_request)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--in_shape', type=int, nargs='+')
    parser.add_argument('--out_shape', type=int, nargs='+')
    parser.add_argument('--model_output', default="semantic_embedding")
    parser.add_argument('--model_configfile')
    parser.add_argument('--model_checkpoint')
    parser.add_argument('--input_dataset_file')
    parser.add_argument('--out_dir')
    parser.add_argument('--out_filename')
    parser.add_argument('--inference_frame', type=int)
    parser.add_argument('--intermediate_layer', type=int)
    parser.add_argument('--dataset_raw_key', default="train/raw")
    parser.add_argument('--dataset_prediction_key', default="train/prediction")
    parser.add_argument('--default_root_dir', default=None)

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
        dataset_prediction_key=options.dataset_prediction_key)
