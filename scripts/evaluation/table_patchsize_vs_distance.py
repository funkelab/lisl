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

def vis_anchor_embedding(embedding, output_file, bg_img=None, subsample=4, scale=4.):
    width, height = embedding.shape[-2:]
    X, Y = np.meshgrid(np.arange(0, width, subsample), np.arange(0, height, subsample))
    U = embedding[0, ::subsample, ::subsample].flatten()
    V = embedding[1, ::subsample, ::subsample].flatten()

    U /= scale
    V /= scale

    if bg_img is not None:
        img = mpimg.imread(bg_img)
        plt.imshow(img)

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

    for i in range(10):
        df = pd.read_csv(f"{root_folder}/score_{i}.csv", header=None)
        scores.append(np.array(df.loc[df[0] == n_samples])[0, 1])

    return np.array(scores)


table = {}

global_path = "/cephfs/swolf/swolf/lisl/experiments/cpc_dsb_38/01_train"
global_path = "/nrs/funke/wolfs2/lisl/experiments/cpc_dsb_01/01_train"
output_file = f"{global_path}/patchsize_vs_distance.csv"
n_samples = 1000
timepoint = "00021999"
scale = 4

os.makedirs(f'{global_path}/img', exist_ok=True)

with open(output_file, "w") as outcsv:
    outcsv.write("patchsize,context_distance,setupfolder,seg_score_mean,seg_score_std,seg_score_min,seg_score_max\n")
    for fn in glob(f"{global_path}/setup_*/evaluation/{timepoint}/embedding_0.h5"):
        setupfolder = fn.split("/")[-4]
        setupscript = "/".join(fn.split("/")[:-3]) + "/train.sh"
        score_root_folder = "/".join(fn.split("/")[:-3]) + f"/evaluation/{timepoint}/"

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

        for idx in [0]:#range(10):
            fc = fn[:-4] + str(idx) + fn[-3:]
            with h5py.File(fc, "r") as fin:
                # print(fin["data"].shape)
                embedding = fin["data"][:, 0]
                vis_anchor_embedding(embedding,
                    f'{global_path}/img/embedding_{setupfolder}_{idx}.png',
                    bg_img=f'{score_root_folder}/{idx}_img_val.png',
                    scale=scale)

                rel_embedding = embedding
                X, Y = np.meshgrid(np.arange(0, rel_embedding.shape[-2]),
                                   np.arange(0, rel_embedding.shape[-1]))
                rel_embedding[0] += X / scale
                rel_embedding[1] += Y / scale

                for c in range(3):
                    imsave(f'{global_path}/img/re_{setupfolder}_{idx}_{c}.png', rel_embedding[c])

                for bandwidth in [None] + list(np.arange(2., 10., 1.)):
                    c, w, h = rel_embedding.shape
                    X = rel_embedding.reshape(c, -1).T
                    print("ms start")
                    ms = MeanShift(bandwidth=bandwidth)
                    ms_seg = ms.fit(X[np.random.rand(len(X)) < 0.01])
                    print("ms fit end")
                    ms_seg = ms.predict(X)
                    print("ms pred end")
                    print(ms_seg.shape)
                    ms_seg = ms_seg.reshape(w, h)
                    colseg = label2color(ms_seg).transpose(1, 2, 0)
                    print(colseg.shape)
                    imsave(f'{global_path}/img/msseg_{setupfolder}_{idx}_{bandwidth}.png', colseg)
                    


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