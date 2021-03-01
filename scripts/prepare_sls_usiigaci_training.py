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
    p.add('-c', '--cleanup', required=False, action='store_true', help='clean up - remove specified train setup')
    p.add('--args', required=False, default="", help='arguments passed to the running script')

    options = p.parse_args()

    experiment_number = 0

    for lr in [4e-5]:
      for patchsize in [16]:
        for regularization in [0.00001, 0.0000001]:
          for dist_factor in [50, 80]:
            for image_scale in [1.]:
              for anchor_radius in [5]:

                out_channels = 2
                stride = 2
                patchdilation = 1
                patch_factor = patchsize//16

                args = options.args + f" --initial_lr {lr}"
                args += f" --out_channels {out_channels}"
                args += f" --patch_size {patchsize}"
                args += f" --patch_overlap {patchsize-stride}"
                args += f" --max_dist {dist_factor}"
                args += f" --dataset UsiigaciDataModule"
                args += f" --regularization {regularization}"
                args += f" --anchor_radius {anchor_radius}"
                args += f" --image_scale {image_scale}"
                args += f" --resnet_size 18"
                args += f" --check_val_every_n_epoch 100"
                args += f" --val_patch_inference_steps 1000"
                # args += f" --distributed_backend 'ddp'"
                args += f" --gpu 1"

                print(f"setting up {options.base_dir} {experiment_number}")
                set_up_experiment(options.base_dir,
                                  options.pybin,
                                  options.experiment_library,
                                  options.script,
                                  options.experiment,
                                  experiment_number,
                                  options.cleanup,
                                  args,
                                  ngpu=1,
                                  ncpu=5)

                experiment_number += 1

