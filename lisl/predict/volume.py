from lisl.gp import (
    AddChannelDim,
    RemoveChannelDim,
    TransposeDims)
import daisy
import gunpowder as gp
import logging

logger = logging.getLogger(__name__)

def predict_volume(
        model,
        dataset,
        out_dir,
        out_filename,
        out_ds_names,
        checkpoint,
        input_name='raw_0',
        normalize_factor=None,
        model_output=0,
        in_shape=None,
        out_shape=None,
        spawn_subprocess=True,
        num_workers=0,
        z_is_time=True):

    raw = gp.ArrayKey('RAW')
    prediction = gp.ArrayKey('PREDICTION')

    data = daisy.open_ds(dataset.filename, dataset.ds_names[0])
    source_roi = gp.Roi(data.roi.get_offset(), data.roi.get_shape())
    voxel_size = gp.Coordinate(data.voxel_size)
    data_dims = len(data.shape)

    # Get in and out shape
    if in_shape is None:
        in_shape = model.in_shape
    if out_shape is None:
        out_shape = model.out_shape

    in_shape = gp.Coordinate(in_shape)
    out_shape = gp.Coordinate(out_shape)
    spatial_dims = in_shape.dims()
    is_2d = spatial_dims == 2

    print(in_shape, out_shape, voxel_size)

    in_shape = in_shape * voxel_size
    out_shape = out_shape * voxel_size

    logger.info(f"source roi: {source_roi}")
    logger.info(f"in_shape: {in_shape}")
    logger.info(f"out_shape: {out_shape}")
    logger.info(f"voxel_size: {voxel_size}")

    request = gp.BatchRequest()
    request.add(raw, in_shape)
    request.add(prediction, out_shape)

    context = (in_shape - out_shape) / 2

    source = (
        gp.ZarrSource(
            dataset.filename,
            {
                raw: dataset.ds_names[0],
            },
            array_specs={
                raw: gp.ArraySpec(
                    roi=source_roi,
                    interpolatable=True)
            }
        )
    )

    # ensure raw has sample and channel dims
    #
    # n = number of samples
    # c = number of channels

    # 2D raw is either (n, y, x) or (c, n, y, x)
    # 3D raw is either (z, y, x) or (c, z, y, x)
    num_additional_channels = (2 + spatial_dims) - data_dims

    for _ in range(num_additional_channels):
        source += AddChannelDim(raw)

    # 2D raw: (c, n, y, x)
    # 3D raw: (c, n=1, z, y, x)

    # prediction requires samples first, channels second
    source += TransposeDims(raw, (1, 0) + tuple(range(2, 2 + spatial_dims)))

    # 2D raw: (n, c, y, x)
    # 3D raw: (n=1, c, z, y, x)

    with gp.build(source):
        raw_roi = source.spec[raw].roi
        logger.info(f"raw_roi: {raw_roi}")

    pipeline = source
        
    if normalize_factor != "skip":
        pipeline = pipeline + gp.Normalize(raw, factor=normalize_factor)

    pipeline = pipeline + (
            gp.Pad(raw, context) +
            gp.torch.Predict(
                model,
                inputs={
                    input_name: raw
                },
                outputs={
                    model_output: prediction
                },
                array_specs={
                    prediction: gp.ArraySpec(roi=raw_roi)
                },
                checkpoint=checkpoint,
                spawn_subprocess=spawn_subprocess
            )
        )

    # 2D raw       : (n, c, y, x)
    # 2D prediction: (n, c, y, x)
    # 3D raw       : (n=1, c, z, y, x)
    # 3D prediction: (n=1, c, z, y, x)

    if is_2d:

        # restore channels first for 2D data
        pipeline += TransposeDims(
            raw,
            (1, 0) + tuple(range(2, 2 + spatial_dims)))
        pipeline += TransposeDims(
            prediction,
            (1, 0) + tuple(range(2, 2 + spatial_dims)))

    else:

        # remove sample dimension for 3D data
        pipeline += RemoveChannelDim(raw)
        pipeline += RemoveChannelDim(prediction)

    # 2D raw       : (c, n, y, x)
    # 2D prediction: (c, n, y, x)
    # 3D raw       : (c, z, y, x)
    # 3D prediction: (c, z, y, x)

    pipeline += (
        gp.ZarrWrite(
            {
                prediction: out_ds_names[0],
            },
            output_dir=out_dir,
            output_filename=out_filename,
            compression_type='gzip') +
        gp.Scan(request, num_workers=num_workers)
    )

    logger.info(
        "Writing prediction to %s/%s[%s]",
        out_dir,
        out_filename,
        out_ds_names[0])

    with gp.build(pipeline):
        pipeline.request_batch(gp.BatchRequest())
