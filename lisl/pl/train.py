import time
import os
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from argparse import ArgumentParser
import pytorch_lightning as pl
from lisl.pl.trainer import SSLTrainer
from lisl.pl.datamodules import MosaicDataModule, SSLDataModule
from lisl.pl.utils import save_args, import_by_string
from lisl.pl.evaluation import SupervisedLinearSegmentationValidation, AnchorSegmentationValidation
from lisl.pl.callbacks import Timing

from pytorch_lightning.callbacks import ModelCheckpoint
from test_tube import HyperOptArgumentParser
import json
# pl.seed_everything(123)


if __name__ == '__main__':

    parser = ArgumentParser()

    parser.add_argument('--dataset', type=str, default="DSBDataModule")
    ds_args = parser.parse_known_args()[0]
    dmodule = import_by_string(f'lisl.pl.datamodules.{ds_args.dataset}')

    print(dmodule)
    
    parser = SSLTrainer.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    parser = dmodule.add_argparse_args(parser)
    parser = dmodule.add_model_specific_args(parser)
    # parser = SupervisedLinearSegmentationValidation.add_model_specific_args(parser)
    args = parser.parse_args()

    # init module
    model = SSLTrainer.from_argparse_args(args)
    datamodule = dmodule.from_argparse_args(args)
    # ssl_test_acc = SupervisedLinearSegmentationValidation.from_argparse_args(args)
    anchor_val = AnchorSegmentationValidation(run_ms_segmentation=False)
    # lr_logger = LearningRateLogger()
    timer = Timing()
    model_saver = ModelCheckpoint(save_last=True, save_weights_only=False, period=10)

    #  init trainer
    trainer = pl.Trainer.from_argparse_args(args)

    # trainer.callbacks.append(ssl_test_acc)
    trainer.callbacks.append(anchor_val)
    # trainer.callbacks.append(lr_logger)
    trainer.callbacks.append(timer)
    trainer.fit(model, datamodule)
