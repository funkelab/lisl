from slurmexperimentmanager.prepare_experiment import set_up_experiment
import configargparse

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

    for lr in [1e-4, 1e-5]:
      for patchsize in [32]:
        for regularization in [0., 0.1, 0.01]:
          for dist_factor in [1, 2, 3]:
            for anchor_radius in [5, 10, 20]:
                hidden_channels = 2
                stride = 9
                patchdilation = 1

                args = options.args + f" --initial_lr {lr}"
                args += f" --hidden_channels {hidden_channels}"
                args += f" --patch_size {patchsize}"
                args += f" --patch_overlap {patchsize-stride}"
                args += f" --max_dist {(patchsize//16)*dist_factor*20}"
                args += f" --regularization {regularization}"
                args += f" --anchor_radius {anchor_radius}"

                print(f"setting up {options.base_dir} {experiment_number}")
                set_up_experiment(options.base_dir,
                                  options.pybin,
                                  options.experiment_library,
                                  options.script,
                                  options.experiment,
                                  experiment_number,
                                  options.cleanup,
                                  args)

                experiment_number += 1

