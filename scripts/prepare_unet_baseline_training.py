from slurmexperimentmanager.prepare_experiment import set_up_experiment
import configargparse
import numpy as np
import math

def run(options, offset, limit, emb_keys, inchannels, ngpu=1):

    args = options.args + f" --ds_file_postfix=.zarr "
    args += f" --augmentations 17 "
    args += f" --batch_size 10 "
    args += f" --gpus 1 "
    args += f" --loader_workers 5 "
    if limit is not None:
        args += f" --ds_limit {offset} {offset + limit} "
    args += f" --max_epochs 100 "
    args += f" --check_val_every_n_epoch 1 "

    args += f" --max_steps 20000 "
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
    # offset_dict = {1:[1 * i for i in range(10)],
    #                2:[2 * i for i in range(8)],
    #                4:[4 * i for i in range(8)],
    #                9:[9 * i for i in range(5)],
    #                18:[18 * i for i in range(3)],
    #                34:[34 * i for i in range(3)],
    #                65:[65 * i for i in range(3)],
    #                124:[0, 124, 124*2],
    #                237:[0, 100, 200],
    #                326:[0, 445-326],
    #                None:[0]}
    max_image_number = 445
    limlist = list(np.unique(np.logspace(0, np.log10(max_image_number), num=8, base=10).astype(int)))
    perm =  np.random.RandomState(seed=42).permutation(max_image_number)
    offset_dict = {None: [0]}
    for l in limlist:
        if 2 * l < max_image_number:
            offset_dict[l] = np.linspace(0, max_image_number - l - 1, num = 3, dtype=int)
        elif l < max_image_number:
            offset_dict[l] = [0, max_image_number-l]
        else:
            offset_dict[l] = [0]

    print("offset_dict", offset_dict)

    keys_and_channels = {"raw": 1,
                      "raw train/prediction cooc_up1.25 cooc_up1.5 cooc_up1.75 cooc_up2.0 cooc_up3.0 cooc_up4.0": 15,
                      "raw simclr": 1+32,
                      "raw train/prediction cooc_up1.25 cooc_up1.5 cooc_up1.75 cooc_up2.0 cooc_up3.0 cooc_up4.0 simclr": 1+14+32}

    for emb_keys, inchannels in keys_and_channels.items():
        for limit in limlist + [None]:
            for offset in offset_dict[limit]:
                run(options, offset, limit, emb_keys, inchannels)
                experiment_number += 1
