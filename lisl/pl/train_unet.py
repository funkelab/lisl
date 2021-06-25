import time
import os
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from argparse import ArgumentParser
import pytorch_lightning as pl
from lisl.pl.trainer_unet import SSLUnetTrainer
from lisl.pl.utils import save_args, import_by_string, SaveModelOnValidation
from lisl.pl.datamodules import ThreeClassDataModule
from lisl.pl.callbacks import Timing

from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from test_tube import HyperOptArgumentParser
import wandb
from pytorch_lightning.loggers import WandbLogger
import json
# pl.seed_everything(123)


if __name__ == '__main__':

    parser = ArgumentParser()

    parser.add_argument('--dataset', type=str, default="ThreeClassDataModule")
    ds_args = parser.parse_known_args()[0]
    dmodule = import_by_string(f'lisl.pl.datamodules.{ds_args.dataset}')

    pl.utilities.seed.seed_everything(42)
    parser = SSLUnetTrainer.add_model_specific_args(parser)

    parser = pl.Trainer.add_argparse_args(parser)
    parser = dmodule.add_argparse_args(parser)
    parser = dmodule.add_model_specific_args(parser)

    args = parser.parse_args()

    # init module
    model = SSLUnetTrainer.from_argparse_args(args)
    datamodule = dmodule.from_argparse_args(args)
    lr_logger = LearningRateMonitor(logging_interval='step')

    checkpoint_callback = ModelCheckpoint(
        monitor='val/f1_0.9',
        dirpath='models',
        filename='unet-{epoch}-{train_loss:.2f}'
    )

    #  init trainer
    logger = WandbLogger(project='ThreeClassTraining', entity='swolf')
    trainer = pl.Trainer.from_argparse_args(args, logger=logger)
    trainer.callbacks.append(checkpoint_callback)
    trainer.callbacks.append(lr_logger)
    trainer.fit(model, datamodule)
