from slurmexperimentmanager.prepare_experiment import set_up_experiment
import configargparse
import numpy as np
import math

def run(options,
        experiment_number,
        frame):

    args = options.args
    args += f" --inference_frame {frame} "

    print(f"setting up {options.base_dir} {experiment_number}")
    set_up_experiment(options.base_dir,
                      options.pybin,
                      options.experiment_library,
                      options.script,
                      options.experiment,
                      experiment_number,
                      experiment_chapter="02_inference",
                      run_script=f"infer_{i}.sh",
                      clean_up=options.cleanup,
                      arguments=args,
                      ngpu=1,
                      ncpu=5,
                      exists_ok=True)


if __name__ == '__main__':

    p = configargparse.ArgParser()
    p.add('-d', '--base_dir', required=False, 
          help='base directory for storing the experiments``')
    p.add('-e', '--experiment', required=True, help='name of the experiment')
    p.add('-r', '--script', required=True, help='script to run')
    p.add('-p', '--pybin', default='python', help='path to python binary')
    p.add('-l', '--experiment_library', help='path to experiment library')
    p.add('-c', '--cleanup', required=False, action='store_true', help='clean up - remove specified train setup')
    p.add('--args', required=False, default="", help='arguments passed to the running script')
    p.add('--max_frames', type=int)

    options = p.parse_args()
    experiment_number = 0

    for i in range(options.max_frames):
        run(options,
            experiment_number,
            frame=i)

