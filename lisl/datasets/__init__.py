import configparser


class Dataset:

    def __init__(self, filename, ds_names):
        self.filename = filename
        self.ds_names = ds_names


def create(config_file):

    config = configparser.ConfigParser()
    config.read(config_file)

    dataset = Dataset(
        eval(config['dataset']['filename']),
        eval(config['dataset']['ds_names']))

    return dataset
