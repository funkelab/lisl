from .contrastive_volume_loss import ContrastiveVolumeLoss  # noqa
import configparser


def create(config_file):

    config = configparser.ConfigParser()
    config.read(config_file)

    loss_type = eval(config['loss']['type'])
    loss_params = eval(config['loss']['params'])

    return loss_type(**loss_params)
