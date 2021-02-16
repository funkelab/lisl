from slurmexperimentmanager.prepare_experiment import set_up_experiment
import configargparse
import numpy as np

def run(lr,
        patchsize,
        regularization,
        dist_factor,
        image_scale,
        anchor_radius,
        out_channels,
        stride):

    args = options.args + f" --initial_lr {lr}"
    args += f" --out_channels {out_channels}"
    args += f" --patch_size {patchsize}"
    args += f" --patch_overlap {patchsize-stride}"
    args += f" --max_dist {dist_factor}"
    args += f" --dataset Bbbc010DataModule"
    args += f" --regularization {regularization}"
    args += f" --anchor_radius {anchor_radius}"
    args += f" --image_scale {image_scale}"
    args += f" --resnet_size 18"
    args += f" --check_val_every_n_epoch 40"
    args += f" --val_patch_inference_steps 2000"
    args += f" --limit_val_batches 4"
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

    base_lr = 4e-5
    base_patchsize = 16
    base_regularization = 0.0001
    base_dist_factor = 30
    base_image_scale = 0.4
    base_anchor_radius = 10
    base_out_channels = 16
    base_stride = 9
    base_patchdilation = 1

    for out_channels in range(2, 16, 2):
        run(base_lr,
            base_patchsize,
            base_regularization,
            base_dist_factor,
            base_image_scale,
            base_anchor_radius,
            out_channels,
            base_stride)
        experiment_number += 1

    # for stride in range(5, 15, 1):
    #     run(base_lr,
    #         base_patchsize,
    #         base_regularization,
    #         base_dist_factor,
    #         base_image_scale,
    #         base_anchor_radius,
    #         base_out_channels,
    #         stride)
    #     experiment_number += 1

    # for regularization in np.logspace(-4, -2.5, base=10., num=10):
    #     run(base_lr,
    #         base_patchsize,
    #         regularization,
    #         base_dist_factor,
    #         base_image_scale,
    #         base_anchor_radius,
    #         base_out_channels,
    #         base_stride)
    #     experiment_number += 1

    # for dist_factor in np.linspace(10, 50, num=10):
    #     run(base_lr,
    #         base_patchsize,
    #         base_regularization,
    #         int(dist_factor),
    #         base_image_scale,
    #         base_anchor_radius,
    #         base_out_channels,
    #         base_stride)
    #     experiment_number += 1


    # for image_scale in np.linspace(1., 3., num=10):
    #     run(base_lr,
    #         base_patchsize,
    #         base_regularization,
    #         base_dist_factor,
    #         image_scale,
    #         base_anchor_radius,
    #         base_out_channels,
    #         base_stride)
    #     experiment_number += 1

    # for anchor_radius in np.linspace(1, 20, num=20):
    #     run(base_lr,
    #         base_patchsize,
    #         base_regularization,
    #         base_dist_factor,
    #         base_image_scale,
    #         int(anchor_radius),
    #         base_out_channels,
    #         base_stride)
    #     experiment_number += 1

    # for _ in range(10):
    #     run(base_lr,
    #         base_patchsize,
    #         base_regularization,
    #         base_dist_factor,
    #         base_image_scale,
    #         base_anchor_radius,
    #         base_out_channels,
    #         base_stride)
    #     experiment_number += 1


