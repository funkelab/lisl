import h5py
from skimage.io import imsave
from glob import glob
import os

import matplotlib
matplotlib.use('agg')
import matplotlib.image as mpimg

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np

from lisl.pl.utils import label2color

from sklearn.cluster import MeanShift

def vis_anchor_embedding(embedding, output_file, bg_img=None, subsample=4, scale=4., image_shape=None):

    if bg_img is not None:
        bg_img = mpimg.imread(bg_img)
        plt.imshow(bg_img)
        width, height = bg_img.shape[-2:]
    else:
        width, height = embedding.shape[-2:]
    
    X, Y = np.meshgrid(np.arange(0, width, subsample), np.arange(0, height, subsample))
    U = embedding[0, ::subsample, ::subsample].flatten()
    V = embedding[1, ::subsample, ::subsample].flatten()

    U /= scale
    V /= scale

    plt.quiver(X, Y, U, V, units='width')

    plt.savefig(output_file, dpi=300)
    plt.cla()
    plt.clf()

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

    for i in range(2):
        df = pd.read_csv(f"{root_folder}/score_{i}.csv", header=None)
        scores.append(np.array(df.loc[df[0] == n_samples])[0, 1])

    return np.array(scores)


table = {}

global_path = "/cephfs/swolf/swolf/lisl/experiments/cpc_dsb_38/01_train"
global_path = "/nrs/funke/wolfs2/lisl/experiments/cpc_dsb_20/01_train"
output_file = f"{global_path}/patchsize_vs_distance.csv"
n_samples = 1000
scale = 1.
n_val_images = 1

img_folder = f"img_emb_evolution"

os.makedirs(f'{global_path}/{img_folder}', exist_ok=True)

for fn in glob(f"{global_path}/setup_*/evaluation/*/embedding_0.h5"):

    setupfolder = fn.split("/")[-4]
    setupscript = "/".join(fn.split("/")[:-3]) + "/train.sh"
    print(fn.split("/"))
    timepoint = fn.split("/")[-2]
    score_root_folder = "/".join(fn.split("/")[:-3]) + f"/evaluation/{timepoint}/"


    config = read_config_file(setupscript)
    patchsize = int(config["patchsize"])
    context_distance = int(config["context_distance"])

    for idx in range(n_val_images):
        fc = fn[:-4] + str(idx) + fn[-3:]
        with h5py.File(fc, "r") as fin:
            # print(fin["data"].shape)
            embedding = fin["data"][:, 0]
            vis_anchor_embedding(embedding,
                f'{global_path}/{img_folder}/embedding_{setupfolder}_{idx}_{timepoint}.png',
                bg_img=f'{score_root_folder}/{idx}_img_val.png',
                scale=scale,
                subsample=1)

