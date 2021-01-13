import h5py
from skimage.io import imsave
from glob import glob

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np

def read_config_file(setupscriptfile):

    with open(setupscriptfile, "r") as script_data:
        return readargs(script_data.readlines()[-1])


def readargs(argstring):

    args = {}
    for s in argstring.split("--"):

        # removes empty strings with if statement
        split = [_ for _ in s.split(" ") if _]

        if len(split) == 2:
            args[split[0]] = split[1]
        elif len(split)> 2:
            args[split[0]] = split[1:]
        else:
            print("ignoring arg ", s)

    return args

def read_seg_scores(root_folder, n_samples=1000):

    scores = []

    for i in range(12):
        df = pd.read_csv(f"{root_folder}/score_{i}.csv", header=None)
        scores.append(np.array(df.loc[df[0] == n_samples])[0, 1])

    return np.array(scores)


table = {}

global_path = "/cephfs/swolf/swolf/lisl/experiments/cpc_dsb_38/01_train"
output_file = f"{global_path}/patchsize_vs_distance.csv"
n_samples = 1000

with open(output_file, "w") as outcsv:
    outcsv.write("patchsize,context_distance,setupfolder,seg_score_mean,seg_score_std,seg_score_min,seg_score_max\n")
    for fn in glob(f"{global_path}/setup_*/evaluation/00027224/embedding_3.h5"):
        with h5py.File(fn, "r") as fin:
            setupfolder = fn.split("/")[-4]
            setupscript = "/".join(fn.split("/")[:-3]) + "/train.sh"
            score_root_folder = "/".join(fn.split("/")[:-3]) + "/evaluation/00027224/"

            config = read_config_file(setupscript)
            patchsize = int(config["patchsize"])
            context_distance = int(config["context_distance"])

            seg_score = read_seg_scores(score_root_folder,
                                        n_samples=n_samples)

            seg_score_mean = seg_score.mean()
            seg_score_std = seg_score.std()
            seg_score_min = seg_score.min()
            seg_score_max = seg_score.max()

            outcsv.write(f"{patchsize},{context_distance},{setupfolder},{seg_score_mean},{seg_score_std},{seg_score_min},{seg_score_max}\n")


with open(output_file, "r") as outcsv:
    data = pd.read_csv(output_file)

    for col in data.columns[3:]:
        pvdata = data.pivot(index='context_distance',
                            columns='patchsize',
                            values=col)
        ax = sb.heatmap(pvdata, annot=True)
        ax.title.set_text(col)
        ax.figure.savefig(f'{global_path}/patchsize_vs_distance_{col}.png')
        ax.cla()
        plt.clf()