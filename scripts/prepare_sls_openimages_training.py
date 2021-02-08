from slurmexperimentmanager.prepare_experiment import set_up_experiment
import configargparse
import numpy as np

if __name__ == '__main__':

    p = configargparse.ArgParser()
    p.add('-d', '--base_dir', required=False, 
          help='base directory for storing the experiments``')
    p.add('-e', '--experiment', required=True, help='name of the experiment, e.g. fafb')
    p.add('-r', '--script', required=True, help='script to run')
    p.add('-p', '--pybin', default='python', help='path to python binary')
    p.add('-l', '--experiment_library', help='path to experiment library')
    p.add('--datamodule', default='DSBDataModule')
    p.add('-c', '--cleanup', required=False, action='store_true', help='clean up - remove specified train setup')
    p.add('--args', required=False, default="", help='arguments passed to the running script')

    options = p.parse_args()

    experiment_number = 0

    for lr in [1e-5]:
      for patchsize in [16]:
        for regularization in [0.1, 0.01, 0.001, 0.0001, 0.00001]:
          for img_scale in [1.]:

            out_channels = 2
            stride = 9
            patchdilation = 1
            patch_factor = patchsize//16
            anchor_radius = 5 * patch_factor
            dist_factor = 20 * patch_factor

            args = options.args + f" --initial_lr {lr}"
            args += f" --out_channels {out_channels}"
            args += f" --patch_size {patchsize}"
            args += f" --dataset {options.datamodule}"
            args += f" --in_channels 3"
            args += f" --gpus 8"
            args += f" --patch_overlap {patchsize-stride}"
            args += f" --max_dist {dist_factor}"
            args += f" --regularization {regularization}"
            args += f" --anchor_radius {anchor_radius}"
            args += f" --image_scale {img_scale}"

            print(f"setting up {options.base_dir} {experiment_number}")
            set_up_experiment(options.base_dir,
                              options.pybin,
                              options.experiment_library,
                              options.script,
                              options.experiment,
                              experiment_number,
                              options.cleanup,
                              args,
                              ngpu=8,
                              ncpu=39)

            experiment_number += 1

