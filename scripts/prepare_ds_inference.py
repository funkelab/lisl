from slurmexperimentmanager.prepare_experiment import set_up_experiment
import configargparse
import numpy as np
import zarr

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
                      experiment_chapter="02_ds_inference",
                      run_script=f"infer_{frame}.sh",
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
    p.add('--max_aug', type=int, default=17)

    options = p.parse_args()
    experiment_number = 0

    inargs = options.args

    for aug in range(options.max_aug):
        print("aug=", aug)
        zarr_input_path = f"/nrs/funke/wolfs2/lisl/datasets/dsb_train_{aug:02}.zarr"
        augargs = inargs + f" --input_dataset_file {zarr_input_path} "
        augargs = augargs + f" --out_filename dsb_train_{aug:02}.zarr "
        augargs = augargs + f" --out_dir /nrs/funke/wolfs2/lisl/datasets "
        augargs = augargs + f" --dataset_raw_key raw "
        options.args = augargs
        inds = zarr.open(zarr_input_path ,"r")
        for frame in inds.keys():
            print("zarr_input_path", zarr_input_path, "frame", frame)
            run(options,
                experiment_number,
                frame=frame)
        experiment_number += 1

    zarr_input_path = f"/nrs/funke/wolfs2/lisl/datasets/fast_dsb_coord_test.zarr"
    augargs = inargs + f" --input_dataset_file {zarr_input_path} "
    augargs = augargs + f" --out_filename fast_dsb_coord_test.zarr "
    augargs = augargs + f" --out_dir /nrs/funke/wolfs2/lisl/datasets "
    augargs = augargs + f" --dataset_raw_key raw "
    options.args = augargs
    inds = zarr.open(zarr_input_path ,"r")
    for frame in inds.keys():
        print("zarr_input_path", zarr_input_path, "frame", frame)
        run(options,
            experiment_number,
            frame=frame)
    experiment_number += 1