from slurmexperimentmanager.prepare_experiment import set_up_experiment
import configargparse
import numpy as np

def run(options,
        lim_images,
        lim_instances_per_image,
        lim_clicks_per_instance,
        initial_temp,
        initial_lr,
        regularization,
        spatial_scaling_factor,
        num_support_tr,
        num_query_tr,
        epochs,
        skip_limits=False):

    experiment_chapter = "03_fast"

    args = options.args
    args += f" --cuda "
    if not skip_limits:
        args += f"--lim_images {lim_images} "
        args += f"--lim_clicks_per_instance {lim_clicks_per_instance} "
        args += f"--lim_instances_per_image {lim_instances_per_image} "

    if options.raw_baseline:
        args += f" --train_on_raw "
        experiment_chapter = "03_baseline"

    args += f'--initial_temp {initial_temp} '
    args += f'--learning_rate {initial_lr} '
    args += f'--regularization {regularization} '
    args += f'--spatial_scaling_factor {spatial_scaling_factor} '

    args += f'--num_support_tr {num_support_tr} '
    args += f'--num_query_tr {num_query_tr} '

    args += f"--epochs {epochs} "
    args += f"--distance_fn rbf "
    args += f"--experiment_root . "

    print(f"setting up {options.base_dir} {experiment_number}")                                           
    set_up_experiment(options.base_dir,
                      options.pybin,
                      options.experiment_library,
                      options.script,
                      options.experiment,
                      experiment_number,
                      experiment_chapter=experiment_chapter,
                      clean_up=options.cleanup,
                      arguments=args,
                      ngpu=1,
                      ncpu=5,
                      queue='gpu_tesla_large')
                    #   queue='gpu_t4')


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
    p.add('--raw_baseline', action='store_true')
    
    options = p.parse_args()

    experiment_number = 0
    base_lim_clicks_per_instance = 8
    num_support_tr = 4
    num_query_tr = 4
    base_epochs = 8
    max_images = 447
    max_instances = 447
    min_lim_instances_per_image = 2
    min_instances_per_image = 2
    min_lim_images = 1
    max_classes_per_it_tr = 40

    steps = 10
    lim = 200
    temp = 8.
    lr = 0.003
    reg = 0.01
    sf = 1.

    instance_limits = np.unique(np.logspace(
        np.log10(2), np.log10(max_instances), base=10, num=steps).astype(int))

    for lim in instance_limits:

        run(options,
            lim,
            lim,
            base_lim_clicks_per_instance,
            temp,
            lr,
            reg,
            sf,
            num_support_tr,
            num_query_tr,
            base_epochs)

        experiment_number += 1
