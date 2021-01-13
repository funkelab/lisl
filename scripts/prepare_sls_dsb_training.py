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

    for lr in [5e-5, 1e-4, 1e-5]:
      for context_distance in [4, 8, 16, 24, 32, 64, 96, 128]:
        for patchsize in [16, 24, 32, 64, 96, 128]:
            #for loss_name in ["anchor"]:#directionclass"]:#"anchor"
        # for patchdilation in [1, 2, 3, 4]:

          # for unet_type, test_out_shape, test_input_name, hidden_channels, bs in zip(["gp", "dunet", "deeplab"], ["120 120", "256 256", "256 256"], ["x", "input_", "x"], [137, 137, 137], [16, 16, 2]):
            loss_name = "anchor"
            unet_type = "resnet"
            test_out_shape = "49, 49"
            test_input_name = "x"
            hidden_channels = 137
            patchoverlap = 5
            patchdilation = 1
            hidden_channels = 3

            args = options.args + f" --initial_lr {lr}"
            args += f" --unet_type {unet_type}"
            args += f" --hidden_channels {hidden_channels}"
            args += f" --context_distance {context_distance}"
            args += f" --patch_size_overlap_dilation {patchsize} {patchoverlap} {patchdilation}"
            args += f" --patchsize {patchsize}"
            args += f" --patchoverlap {patchoverlap}"
            args += f" --patchdilation {patchdilation}"
            args += f" --loss_name {loss_name}"

            set_up_experiment(options.base_dir,
                              options.pybin,
                              options.experiment_library,
                              options.script,
                              options.experiment,
                              experiment_number,
                              options.cleanup,
                              args)

            experiment_number += 1

