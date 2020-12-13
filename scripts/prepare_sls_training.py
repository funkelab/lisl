from slurmexperimentmanager.prepare_experiment import set_up_experiment
import configargparse

if __name__ == '__main__':

    p = configargparse.ArgParser()
    p.add('-d', '--base_dir', required=False, 
          help='base directory for storing micron experiments``')
    p.add('-e', '--experiment', required=True, help='name of the experiment, e.g. fafb')
    p.add('-r', '--script', required=True, help='script to run')
    p.add('-p', '--pybin', default='python', help='path to python binary')
    p.add('-l', '--experiment_library', help='path to experiment library')
    p.add('-c', '--cleanup', required=False, action='store_true', help='clean up - remove specified train setup')
    p.add('--args', required=False, default="", help='arguments passed to the running script')

    options = p.parse_args()

    experiment_number = 0

    for loss in ["spatialconsensus"]:

        for lr in [1e-4, 1e-5]:

            for distance in [2, 4, 8, 16]:

              options.args =  options.args + f" --loss_name {loss}" + f" --initial_lr {lr} --distance {distance}"

              set_up_experiment(options.base_dir,
                                options.pybin,
                                options.experiment_library,
                                options.script,
                                options.experiment,
                                experiment_number,
                                options.cleanup,
                                options.args)

              experiment_number += 1

