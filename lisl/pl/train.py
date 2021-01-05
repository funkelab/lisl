import time
import os
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from argparse import ArgumentParser
import pytorch_lightning as pl
from lisl.pl.trainer import SSLTrainer, ContextPredictionTrainer
from lisl.pl.datamodules import MosaicDataModule, SSLDataModule, DSBDataModule
from lisl.pl.utils import save_args
from lisl.pl.evaluation import SupervisedLinearSegmentationValidation

from pytorch_lightning.callbacks import ModelCheckpoint
from test_tube import HyperOptArgumentParser
import json
# pl.seed_everything(123)

from pytorch_lightning.callbacks import Callback
from time import time

class Timing(Callback):

    def setup(self, trainer, pl_module, stage: str, log_memory=True):
        """Called when fit or test begins"""
        self.last_state = None
        self.last_time  = None

        self.last_batch_time  = None
        self.log_memory = log_memory


    def teardown(self, trainer, pl_module, stage: str):
        """Called when fit or test ends"""
        pass        

    def get_memory_stats(self, pl_module):
        if self.log_memory:
            pl_module.logger.log_metrics({"reserved_bytes.all.current": 
                torch.cuda.memory_stats()["reserved_bytes.all.current"] / 2**30},
                step=pl_module.global_step)
            pl_module.logger.log_metrics({"reserved_bytes.all.peak": 
                torch.cuda.memory_stats()["reserved_bytes.all.peak"] / 2**30},
                step=pl_module.global_step)
            

    def on_train_batch_start(self, trainer, pl_module, *args):
        """Called when the train batch begins."""
        newtime = time()
        if self.last_state is not None:
            timediff = newtime - self.last_time
            pl_module.logger.log_metrics({f"{self.last_state}_to_train_batch_start": timediff}, step=pl_module.global_step)
            
        self.last_time = newtime
        self.last_state = "train_batch_start"

        if self.last_batch_time is not None:
            timediff = newtime - self.last_batch_time
            pl_module.logger.log_metrics({"batch_to_batch_time": timediff}, step=pl_module.global_step)
            
        self.last_batch_time = newtime
        self.get_memory_stats(pl_module)



    def on_train_batch_end(self, trainer, pl_module, *args):
        """Called when the train batch ends."""
        newtime = time()
        if self.last_state is not None:
            timediff = newtime - self.last_time
            pl_module.logger.log_metrics({f"{self.last_state}_to_train_batch_end": timediff}, step=pl_module.global_step)
            
        self.last_time = newtime
        self.last_state = "train_batch_end"
        self.get_memory_stats(pl_module)

    def on_train_epoch_start(self, trainer, pl_module, *args):
        """Called when the train epoch begins."""
        newtime = time()
        if self.last_state is not None:
            timediff = newtime - self.last_time
            pl_module.logger.log_metrics({f"{self.last_state}_to_train_epoch_start": timediff}, step=pl_module.global_step)
            
        self.last_time = newtime
        self.last_state = "train_epoch_start"
        self.get_memory_stats(pl_module)

    def on_train_epoch_end(self, trainer, pl_module, *args):
        """Called when the train epoch ends."""
        newtime = time()
        if self.last_state is not None:
            timediff = newtime - self.last_time
            pl_module.logger.log_metrics({f"{self.last_state}_to_train_epoch_end": timediff}, step=pl_module.global_step)
            
        self.last_time = newtime
        self.last_state = "train_epoch_end"
        self.get_memory_stats(pl_module)


    def on_batch_end(self, trainer, pl_module, *args):
        """Called when the training batch ends."""
        newtime = time()
        if self.last_state is not None:
            timediff = newtime - self.last_time
            pl_module.logger.log_metrics({f"{self.last_state}_to_batch_end": timediff}, step=pl_module.global_step)
            
        self.last_time = newtime
        self.last_state = "batch_end"
        self.get_memory_stats(pl_module)


if __name__ == '__main__':

    parser = ArgumentParser()
    parser = ContextPredictionTrainer.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    parser = DSBDataModule.add_argparse_args(parser)
    parser = DSBDataModule.add_model_specific_args(parser)
    # parser = SupervisedLinearSegmentationValidation.add_model_specific_args(parser)

    args = parser.parse_args()

    # init module
    model = ContextPredictionTrainer.from_argparse_args(args)
    datamodule = DSBDataModule.from_argparse_args(args)
    # ssl_test_acc = SupervisedLinearSegmentationValidation.from_argparse_args(args)
    # lr_logger = LearningRateLogger()
    timer = Timing()
    model_saver = ModelCheckpoint(save_last=True, save_weights_only=False, period=10)

    #  init trainer
    trainer = pl.Trainer.from_argparse_args(args)

    # trainer.callbacks.append(ssl_test_acc)
    # trainer.callbacks.append(lr_logger)
    trainer.callbacks.append(timer)
    trainer.fit(model, datamodule)
