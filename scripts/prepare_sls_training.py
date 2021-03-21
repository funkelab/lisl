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

    for lr in [1e-3]:
      for context_distance in [32, 64, 128, 256]:

        for unet_type, test_out_shape, test_input_name, hidden_channels in zip(["gp", "dunet", "deeplab"], ["120 120", "256 256", "256 256"], ["x", "input_", "x"], [128, 485, 128]):

            args =  options.args + f" --initial_lr {lr} --unet_type {unet_type} --test_out_shape {test_out_shape} --test_input_name {test_input_name} --hidden_channels {hidden_channels} --context_distance {context_distance}"

            set_up_experiment(options.base_dir,
                              options.pybin,
                              options.experiment_library,
                              options.script,
                              options.experiment,
                              experiment_number,
                              experiment_chapter="01_train",
                              clean_up=options.cleanup,
                              arguments=args)

            experiment_number += 1

