from lisl.gp import Blur
import configparser
import gunpowder as gp
import math  # noqa


def create(config_file):

    config = configparser.ConfigParser()
    config.read(config_file)

    params = {
        k: eval(v)
        for k, v in config['augmentation'].items()
    }

    return params


def add_augmentation_pipeline(
        pipeline,
        raw,
        simple=None,
        elastic=None,
        blur=None,
        noise=None):
    '''Add an augmentation pipeline to an existing pipeline.

    All optional arguments are kwargs for the corresponding augmentation node.
    If not given, those augmentations are not added.
    '''

    if simple is not None:
        pipeline = pipeline + gp.SimpleAugment(**simple)

    if elastic is not None:
        pipeline = pipeline + gp.ElasticAugment(**elastic)

    if blur is not None:
        pipeline = pipeline + Blur(raw, **blur)

    if noise is not None:
        pipeline = pipeline + gp.NoiseAugment(raw, **noise)

    return pipeline
