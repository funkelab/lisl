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
        input_key='0/raw',
        normalize_factor=None,
        model_output=0,
        in_shape=None,
        out_shape=None,
        spawn_subprocess=True,
        num_workers=0):

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

    if apply_voxel_size:
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

    print("context", context, in_shape, out_shape)

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

    num_additional_channels = (2 + spatial_dims) - data_dims

    for _ in range(num_additional_channels):
        source += AddChannelDim(raw)

    # prediction requires samples first, channels second
    source += TransposeDims(raw, (1, 0) + tuple(range(2, 2 + spatial_dims)))

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

    # # remove sample dimension for 3D data
    # pipeline += RemoveChannelDim(raw)
    # pipeline += RemoveChannelDim(prediction)

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

    with gp.build(pipeline):
        pipeline.request_batch(gp.BatchRequest())




    # raw_0 = gp.ArrayKey('RAW_0')
    # emb_0 = gp.ArrayKey('EMBEDDING_0')

    # data = daisy.open_ds(dataset, input_key)
    # source_roi = gp.Roi(data.roi.get_offset(), data.roi.get_shape())
    # # voxel_size = gp.Coordinate(data.voxel_size)
    # # emb_voxel_size = voxel_size

    # # Get in and out shape
    # in_shape = gp.Coordinate(model.in_shape)
    # out_shape = gp.Coordinate(model.out_shape)

    # logger.info(f"source roi: {source_roi}")
    # logger.info(f"in_shape: {in_shape}")
    # logger.info(f"out_shape: {out_shape}")
    # logger.info(f"voxel_size: {voxel_size}")

    # request = gp.BatchRequest()
    # request.add(raw_0, in_shape)
    # request.add(raw_1, in_shape)
    # request.add(points_0, out_shape)
    # request.add(points_1, out_shape)
    # request[locations_0] = gp.ArraySpec(nonspatial=True)
    # request[locations_1] = gp.ArraySpec(nonspatial=True)

    # snapshot_request = gp.BatchRequest()
    # snapshot_request[emb_0] = gp.ArraySpec(roi=request[points_0].roi)
    # snapshot_request[emb_1] = gp.ArraySpec(roi=request[points_1].roi)

    # # Let's hardcode this for now
    # # TODO read actual number from zarr file keys
    # n_samples = 447
    # batch_size = 1
    # dim = 2
    # padding = (100, 100)

    # sources = []
    # for i in range(n_samples):

    #     ds_key = f'train/raw/{i}'
    #     image_sources = tuple(
    #         gp.ZarrSource(
    #             dataset.filename,
    #             {
    #                 raw: ds_key
    #             },
    #             {
    #                 raw: gp.ArraySpec(interpolatable=True, voxel_size=(1, 1))
    #             }) + gp.Pad(raw, None)
    #         for raw in [raw_0, raw_1])

    #     random_point_generator = RandomPointGenerator(
    #         density=point_density, repetitions=2)

    #     point_sources = tuple((
    #         RandomPointSource(
    #             points_0,
    #             dim,
    #             random_point_generator=random_point_generator),
    #         RandomPointSource(
    #             points_1,
    #             dim,
    #             random_point_generator=random_point_generator)
    #     ))

    #     # TODO: get augmentation parameters from some config file!
    #     points_and_image_sources = tuple(
    #         (img_source, point_source) + gp.MergeProvider() +
    #         gp.SimpleAugment() +
    #         gp.ElasticAugment(
    #             spatial_dims=2,
    #             control_point_spacing=(10, 10),
    #             jitter_sigma=(0.0, 0.0),
    #             rotation_interval=(0, math.pi/2)) +
    #         gp.IntensityAugment(r,
    #                             scale_min=0.8,
    #                             scale_max=1.2,
    #                             shift_min=-0.2,
    #                             shift_max=0.2,
    #                             clip=False) +
    #         gp.NoiseAugment(r, var=0.01, clip=False)
    #         for r, img_source, point_source
    #         in zip([raw_0, raw_1], image_sources, point_sources))

    #     sample_source = points_and_image_sources + gp.MergeProvider()

    #     data = daisy.open_ds(dataset.filename, ds_key)
    #     source_roi = gp.Roi(data.roi.get_offset(), data.roi.get_shape())
    #     sample_source += gp.Crop(raw_0, source_roi)
    #     sample_source += gp.Crop(raw_1, source_roi)
    #     sample_source += gp.Pad(raw_0, padding)
    #     sample_source += gp.Pad(raw_1, padding)
    #     sample_source += gp.RandomLocation()
    #     sources.append(sample_source)

    # sources = tuple(sources)

    # pipeline = sources + gp.RandomProvider()
    # pipeline += gp.Unsqueeze([raw_0, raw_1])

    # pipeline += PrepareBatch(raw_0, raw_1,
    #                          points_0, points_1,
    #                          locations_0, locations_1)

    # # How does prepare batch relate to Stack?????
    # pipeline += RejectArray(ensure_nonempty=locations_1)
    # pipeline += RejectArray(ensure_nonempty=locations_0)

    # # batch content
    # # raw_0:          (1, h, w)
    # # raw_1:          (1, h, w)
    # # locations_0:    (n, 2)
    # # locations_1:    (n, 2)

    # pipeline += gp.Stack(batch_size)

    # # batch content
    # # raw_0:          (b, 1, h, w)
    # # raw_1:          (b, 1, h, w)
    # # locations_0:    (b, n, 2)
    # # locations_1:    (b, n, 2)

    # pipeline += gp.PreCache(num_workers=10)

    # pipeline += gp.torch.Train(model, loss, optimizer,
    #                            inputs={
    #                                'raw_0': raw_0,
    #                                'raw_1': raw_1
    #                            },
    #                            loss_inputs={
    #                                'emb_0': emb_0,
    #                                'emb_1': emb_1,
    #                                'locations_0': locations_0,
    #                                'locations_1': locations_1
    #                            },
    #                            outputs={
    #                                2: emb_0,
    #                                3: emb_1
    #                            },
    #                            array_specs={
    #                                emb_0: gp.ArraySpec(voxel_size=emb_voxel_size),
    #                                emb_1: gp.ArraySpec(
    #                                    voxel_size=emb_voxel_size)
    #                            },
    #                            checkpoint_basename=os.path.join(
    #                                out_dir, 'model'),
    #                            save_every=checkpoint_interval)

    # pipeline += gp.Snapshot({
    #     raw_0: 'raw_0',
    #     raw_1: 'raw_1',
    #     emb_0: 'emb_0',
    #     emb_1: 'emb_1',
    #     # locations_0 : 'locations_0',
    #     # locations_1 : 'locations_1',
    # },
    #     every=snapshot_interval,
    #     additional_request=snapshot_request)




























