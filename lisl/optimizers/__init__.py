import configparser
import torch  # noqa


def create(model, config_file):

    config = configparser.ConfigParser()
    config.read(config_file)

    optimizer_type = eval('torch.optim.' + config['optimizer']['type'])
    optimizer_params = eval(config['optimizer']['params'])
    num_iterations = eval(config['optimizer']['num_iterations'])

    optimizer = optimizer_type(model.parameters(), **optimizer_params)

    return optimizer, num_iterations
