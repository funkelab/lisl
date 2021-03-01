from slurmexperimentmanager.prepare_experiment import set_up_experiment
import configargparse
import numpy as np
import math

def run(lr,
        patchsize,
        regularization,
        positive_radius,
        image_scale,
        out_channels,
        stride,
        temperature,
        temperature_decay):

    args = options.args + f" --initial_lr {lr}"
    args += f" --out_channels {out_channels}"
    args += f" --patch_size {patchsize}"
    args += f" --patch_overlap {patchsize-stride}"
    args += f" --positive_radius {positive_radius}"
    args += f" --dataset DSBDataModule"
    args += f" --regularization {regularization}"
    args += f" --image_scale {image_scale}"
    args += f" --resnet_size 18"
    args += f" --check_val_every_n_epoch 50"
    args += f" --max_epochs 200"
    args += f" --temperature {temperature}"
    args += f" --temperature_decay {temperature_decay}"
    args += f" --val_patch_inference_steps 2000"
    args += f" --val_patch_inference_downsample 8"
    # args += f" --limit_val_batches 16"
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

    # for lr in [4e-5]:
    #   for patchsize in [16]:
    #     for regularization in [0.001, 0.0001]:
    #       for dist_factor in [30]:
    #         for image_scale in [1.]:
    #           for anchor_radius in [2, 5, 10]:

    base_lr = 1e-6
    base_patchsize = 16
    base_regularization = 0.
    base_dist_factor = 10
    base_image_scale = 1.
    base_anchor_radius = 6
    base_temperature = 10.
    base_out_channels = 2
    base_stride = 3
    base_patchdilation = 1

    temperature_decay = 1.
    regularization = 0.
    temperature = 1.66
    positive_radius = 10
    # for temperature_decay in [math.pow(0.5, 1/halftime) for halftime in np.logspace(2, 6, base=10., num = 3)] + [1.]:
    for image_scale in [1., 2]:
        for regularization in [1., 0.001, 0.00001]:#np.logspace(-7, -1., base=10., num=3):
            for temperature in [1., 2., 4]:
                # for positive_radius in np.linspace(5, 20, num=10):
                run(lr=base_lr,
                    patchsize=base_patchsize,
                    regularization=regularization,
                    positive_radius=int(positive_radius),
                    image_scale=image_scale,
                    out_channels=base_out_channels,
                    stride=base_stride,
                    temperature=temperature,
                    temperature_decay=temperature_decay)
                experiment_number += 1


    #     run(base_lr,
    #         base_patchsize,
    #         regularization,
    #         base_dist_factor,
    #         base_image_scale,
    #         base_anchor_radius,
    #         base_out_channels,
    #         base_stride)
    #     experiment_number += 1

    # for dist_factor in np.linspace(8, 30, num=10):
    #     run(lr=base_lr,
    #         patchsize=base_patchsize,
    #         regularization=base_regularization,
    #         positive_radius=int(dist_factor),
    #         image_scale=base_image_scale,
    #         temperature=base_anchor_radius,
    #         out_channels=base_out_channels,
    #         stride=base_stride)
    #     experiment_number += 1

    # for image_scale in np.linspace(0.4, 2., num=10):
    #     run(lr=base_lr,
    #         patchsize=base_patchsize,
    #         regularization=base_regularization,
    #         positive_radius=base_dist_factor,
    #         image_scale=image_scale,
    #         temperature=base_anchor_radius,
    #         out_channels=base_out_channels,
    #         stride=base_stride)
    #     experiment_number += 1

    # for temperature in np.linspace(5, 20, num=20):
    #     run(lr=base_lr,
    #         patchsize=base_patchsize,
    #         regularization=base_regularization,
    #         positive_radius=base_dist_factor,
    #         image_scale=base_image_scale,
    #         temperature=int(temperature),
    #         out_channels=base_out_channels,
    #         stride=base_stride)
    #     experiment_number += 1

    # for _ in range(10):
    #     run(lr=base_lr,
    #         patchsize=base_patchsize,
    #         regularization=base_regularization,
    #         positive_radius=base_dist_factor,
    #         image_scale=base_image_scale,
    #         temperature=base_anchor_radius,
    #         out_channels=base_out_channels,
    #         stride=base_stride)
    #     experiment_number += 1


