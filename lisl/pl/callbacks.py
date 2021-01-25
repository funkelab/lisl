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
            try:
                pl_module.logger.log_metrics({"reserved_bytes.all.current": 
                    torch.cuda.memory_stats()["reserved_bytes.all.current"] / 2**30},
                    step=pl_module.global_step)
                pl_module.logger.log_metrics({"reserved_bytes.all.peak": 
                    torch.cuda.memory_stats()["reserved_bytes.all.peak"] / 2**30},
                    step=pl_module.global_step)
            except:
                pass
            

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
