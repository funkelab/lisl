from slurmexperimentmanager.prepare_experiment import set_up_experiment
import configargparse
import numpy as np
import math

def run(options,
        data_module,
        lr,
        patchsize,
        regularization,
        positive_radius,
        image_scale,
        out_channels,
        stride,
        # train_time_augmentation,
        temperature,
        temperature_decay,
        check_val_every_n_epoch,
        max_epochs,
        resnet_size,
        batch_size,
        ngpu):

    args = options.args + f" --initial_lr {lr}"
    args += f" --out_channels {out_channels}"
    args += f" --patch_size {patchsize}"
    args += f" --patch_overlap {patchsize-stride}"
    args += f" --positive_radius {positive_radius}"
    args += f" --dataset {data_module}"
    # args += f" --image_folder {image_folder}"
    # args += f" --annotation_file {annotation_file}"
    # args += f" --test_annotation_file {test_annotation_file}"
    args += f" --regularization {regularization}"
    args += f" --image_scale {image_scale}"
    args += f" --resnet_size {resnet_size}"
    args += f" --check_val_every_n_epoch {check_val_every_n_epoch}"
    args += f" --max_epochs {max_epochs}"
    args += f" --temperature {temperature}"
    args += f" --temperature_decay {temperature_decay}"
    args += f" --val_patch_inference_steps 200 "
    args += f" --lr_milestones 25 50 75 90 100"
    args += f" --batch_size {batch_size}"
    args += f" --loader_workers {8*ngpu}"
    args += f" --gpu {ngpu}"

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
                      ncpu=11*ngpu,
                      queue='gpu_tesla_large')
                    #   queue='gpu_rtx')


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

    base_lr = 1e-6
    base_patchsize = 16
    base_regularization = .02
    base_image_scale = 1.
    base_anchor_radius = 6
    base_temperature = 10.
    base_out_channels = 2
    base_stride = 2
    base_patchdilation = 1

    base_temperature = 10.
    base_temperature_decay = 1.
    base_positive_radius = 10
    ngpu = 4
    batch_size = 12
    check_val_every_n_epoch = 25
    max_epochs = 1200
    dsmodule = "LiveCellDataModule"
    resnet_size = 101

    # for dspath in ["/nrs/funke/wolfs2/lisl/datasets/collection", "/nrs/funke/wolfs2/lisl/datasets/sim"]:
    #     run(options=options,
    #         data_module=dsmodule,
    #         dspath=dspath,
    #         lr=base_lr,
    #         patchsize=base_patchsize,
    #         regularization=base_regularization,
    #         positive_radius=int(base_positive_radius),
    #         image_scale=base_image_scale,
    #         out_channels=2,
    #         stride=base_stride,
    #         # train_time_augmentation="nothing",
    #         temperature=base_temperature,
    #         temperature_decay=base_temperature_decay,
    #         check_val_every_n_epoch=check_val_every_n_epoch,
    #         max_epochs=max_epochs,
    #         resnet_size=resnet_size,
    #         batch_size=batch_size,
    #         ngpu=ngpu)
    #     experiment_number += 1

    run(options=options,
        data_module=dsmodule,
        lr=base_lr,
        patchsize=base_patchsize,
        regularization=base_regularization,
        positive_radius=int(base_positive_radius),
        image_scale=base_image_scale,
        out_channels=2,
        stride=base_stride,
        # train_time_augmentation="nothing",
        temperature=base_temperature,
        temperature_decay=base_temperature_decay,
        check_val_every_n_epoch=check_val_every_n_epoch,
        max_epochs=max_epochs,
        resnet_size=resnet_size,
        batch_size=batch_size,
        ngpu=ngpu)

        # for temperature in np.linspace(1., 20, num=10):
        #     run(data_module=dsmodule,
        #         dspath=dspath,
        #         lr=base_lr,
        #         patchsize=base_patchsize,
        #         regularization=regularization,
        #         positive_radius=int(base_positive_radius),
        #         image_scale=base_image_scale,
        #         out_channels=2,
        #         stride=base_stride,
        #         # train_time_augmentation="nothing",
        #         temperature=temperature,
        #         temperature_decay=base_temperature_decay,
        #         check_val_every_n_epoch=check_val_every_n_epoch,
        #         max_epochs=max_epochs)
        #     experiment_number += 1

        # for positive_radius in np.linspace(5, 30, num=10):

        #     run(data_module=dsmodule,
        #         dspath=dspath,
        #         lr=base_lr,
        #         patchsize=base_patchsize,
        #         regularization=regularization,
        #         positive_radius=int(positive_radius),
        #         image_scale=base_image_scale,
        #         out_channels=2,
        #         stride=base_stride,
        #         # train_time_augmentation="nothing",
        #         temperature=base_temperature,
        #         temperature_decay=base_temperature_decay,
        #         check_val_every_n_epoch=check_val_every_n_epoch,
        #         max_epochs=max_epochs)
        #     experiment_number += 1

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


