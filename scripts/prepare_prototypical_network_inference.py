from slurmexperimentmanager.prepare_experiment import set_up_experiment
import configargparse
import numpy as np
from glob import glob

def run(options,
        experiment_number,
        model_path,
        experimentname,
        zarr_in_path,
        zarr_out_path,
        args):

    args += f" --model_path {model_path} "
    args += f" --experimentname {experimentname} "
    args += f" --zarr_in_path {zarr_in_path} "
    args += f" --zarr_out_path {zarr_out_path} "

    set_up_experiment(options.base_dir,
                      options.pybin,
                      options.experiment_library,
                      options.script,
                      options.experiment,
                      experiment_number,
                      run_script="infer.sh",
                      experiment_chapter="04_pn_inference",
                      clean_up=options.cleanup,
                      arguments=args,
                      ngpu=1,
                      ncpu=1)

    print(f"setting up {options.base_dir} {options.experiment} {experiment_number}")                                           

if __name__ == '__main__':

    p = configargparse.ArgParser()
    p.add('-d', '--base_dir', required=False, 
          help='base directory for storing the experiments``')
    p.add('-e', '--experiment', required=True, help='name of the experiment, e.g. fafb')
    p.add('-r', '--script', required=True, help='script to run')
    p.add('-p', '--pybin', default='python', help='path to python binary')
    p.add('-l', '--experiment_library', help='path to experiment library')
    p.add('-c', '--cleanup', required=False, action='store_true', help='clean up - remove specified train setup')

    p.add('--setuppattern', required=True)
    p.add('--datasetfile', required=True)
    p.add('--outputfile', required=True)
    p.add('--args', required=True)
    options = p.parse_args()

    experiment_number = 0
    for setup_dir in glob(options.setuppattern):
        model_path = f"{setup_dir}/last_model.pth"
        sp = setup_dir.split("/")
        experimentname = f"{sp[-3]}_{sp[-1]}"
        print(f"setup dir {setup_dir}, experimentname {experimentname}")
        zarr_in_path = options.datasetfile
        zarr_out_path = options.outputfile
        run(options,
            experiment_number,
            model_path=model_path,
            experimentname=experimentname,
            zarr_in_path=zarr_in_path,
            zarr_out_path=zarr_out_path,
            args=options.args)
        experiment_number += 1

# example calls
# python prototypical_networks_predict_and_segment.py --experimentname dev00 --model_path /nrs/funke/wolfs2/lisl/experiments/pn_dsb_01/03_fast/setup_t0064/last_model.pth --zarr_in_path /nrs/funke/wolfs2/lisl/datasets/fast_dsb_img_test.zarr --zarr_out_path /nrs/funke/wolfs2/lisl/datasets/fast_dsb_img_test_out.zarr --in_channels 545 --inst_out_channels 4  --n_sem_classes 2
# python scripts/prepare_prototypical_network_inference.py --base_dir /nrs/funke/wolfs2/lisl/experiments -e pn_dsb_03 -r lisl/evaluation/prototypical_networks_predict_and_segment.py -p /groups/funke/home/wolfs2/miniconda3/envs/pytorch/bin/python -l ~/local/src/lisl/lisl --setuppattern "/nrs/funke/wolfs2/lisl/experiments/pn_dsb_03/03_fast/setup_t*" --datasetfile /nrs/funke/wolfs2/lisl/datasets/fast_dsb_img_test.zarr --args "--in_channels 545 --inst_out_channels 4  --n_sem_classes 2"
