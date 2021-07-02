from slurmexperimentmanager.prepare_experiment import set_up_experiment
import configargparse
import numpy as np
import math

def run(options, limit, emb_keys, inchannels, ngpu=1):
    args = options.args + f" --ds_file_postfix=.zarr "
    args += f" --augmentations 17 "
    args += f" --batch_size 10 "
    args += f" --gpus 1 "
    args += f" --loader_workers 10 "
    if limit is not None:
        args += f" --max_epochs {int(100 * (450 / limit))} "
        args += f" --ds_limit {limit} "
        args += f" --check_val_every_n_epoch {int(450 / limit)} "
    else:
        args += f" --max_epochs 100 "
        args += f" --check_val_every_n_epoch 1 "

    args += f" --emb_keys {emb_keys} "
    args += f" --in_channels {inchannels} "
    

    print(f"setting up {options.base_dir} {experiment_number}")
    set_up_experiment(options.base_dir,
                      options.pybin,
                      options.experiment_library,
                      options.script,
                      options.experiment,
                      experiment_number,
                      experiment_chapter="01_train",
                      run_script="train.sh",
                      clean_up=options.cleanup,
                      arguments=args,
                      ngpu=ngpu,
                      ncpu=5*ngpu)


if __name__ == '__main__':

    p = configargparse.ArgParser()
    p.add('-d', '--base_dir', required=False, 
          help='base directory for storing the experiments``')
    p.add('-e', '--experiment', required=True, help='name of the experiment, e.g. fafb')
    p.add('-r', '--script', required=True, help='script to run')
    p.add('-p', '--pybin', default='python', help='path to python binary')
    p.add('-l', '--experiment_library', help='path to experiment library')
    p.add('-c', '--cleanup', required=False, action='store_true', help='clean up - remove specified train setup')
    p.add('--args', required=False, default="", help='arguments passed to the running script')

    options = p.parse_args()
    experiment_number = 0

    for emb_keys, inchannels in zip(["raw train/prediction"], [3]):
        for limit in [1, 2, 3, 4, 6, 9, 13, 18, 24, 34, 47, 65, 90, 124, 171, 237, 326, None]:

            run(options, limit, emb_keys, inchannels)
            experiment_number += 1
