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
    help="The prediction config file to use.")

if __name__ == "__main__":

    options = parser.parse_args()
    config_file = options.config_file

    dataset = lisl.datasets.create(config_file)
    model = lisl.models.create(config_file)
    model.eval()

    config = configparser.ConfigParser()
    config.read(config_file)

    checkpoint = eval(config['model']['checkpoint'])
    raw_normalize = eval(config['predict']['raw_normalize'])
    out_filename = eval(config['predict']['out_filename'])
    out_ds_names = eval(config['predict']['out_ds_names'])
    out_dir = '.'.join(config_file.split('.')[:-1])

    lisl.predict.predict_volume(
        model,
        dataset,
        out_dir,
        out_filename,
        out_ds_names,
        checkpoint,
        raw_normalize)
