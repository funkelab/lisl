from .unet import UNet  # noqa
from .dense_projection_net import DenseProjectionNet  # noqa
import configparser


def create(config_file):

    config = configparser.ConfigParser()
    config.read(config_file)

    model_type = eval(config['model']['type'])

    base_encoder_type = eval(config['model']['base_encoder'])
    base_encoder_params = eval(config['model']['base_encoder_params'])
    base_encoder = base_encoder_type(**base_encoder_params)

    model = model_type(
        base_encoder,
        eval(config['model']['h_channels']),
        eval(config['model']['out_channels']))

    return model
