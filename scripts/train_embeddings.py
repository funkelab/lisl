from tqdm import tqdm
import argparse
import configparser
import gunpowder as gp
import lisl
import torch
import logging

logging.basicConfig(level=logging.INFO)

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser()
parser.add_argument(
    '-c', '--config-file',
    help="The training config file to use.")

if __name__ == "__main__":

    options = parser.parse_args()
    config_file = options.config_file

    dataset = lisl.datasets.create(config_file)
    model = lisl.models.create(config_file)
    loss = lisl.losses.create(config_file)
    optimizer, num_iterations = lisl.optimizers.create(
        model,
        config_file)
    augmentation_parameters = lisl.train.augmentations.create(config_file)

    config = configparser.ConfigParser()
    config.read(config_file)

    point_density = eval(config['train']['point_density'])
    checkpoint_interval = eval(config['train']['checkpoint_interval'])
    snapshot_interval = eval(config['train']['snapshot_interval'])
    raw_normalize = eval(config['train']['raw_normalize'])
    out_dir = eval(config['output']['out_dir'])
    print(out_dir)

    pipeline, request = lisl.train.random_point_pairs_pipeline(
        model, loss, optimizer,
        dataset,
        augmentation_parameters,
        point_density,
        out_dir,
        raw_normalize,
        checkpoint_interval,
        snapshot_interval)

    with gp.build(pipeline):
        for i in tqdm(range(num_iterations)):
            pipeline.request_batch(request)

    print(options)
