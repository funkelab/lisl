from ctcmetrics.seg import seg_metric
import zarr
from tqdm import tqdm
import numpy as np
import pandas as pd
from EmbedSeg.utils.utils2 import matching

expname = "pn_dsb_04"
expname = "pn_dsb_dev16"
expname = "pn_dsb_05"
expname = "pn_dsb_22_bgzero"

base_folder = f"/nrs/funke/wolfs2/lisl/experiments/{expname}/03_fast/"
prediction_file = "/nrs/funke/wolfs2/lisl/datasets/fast_dsb_coord_test_inference.zarr"
gt_file = "/nrs/funke/wolfs2/lisl/datasets/fast_dsb_coord_test.zarr"

pred_zarr = zarr.open(prediction_file, "r")
gt_zarr = zarr.open(gt_file, "r")
outfile = f"/nrs/funke/wolfs2/lisl/experiments/{expname}/labelvsscore.csv"

# with open(outfile, "w") as csv_file:
scores_names = []
for postfix in ["", "_foreground=gt"]:
    for bw in [3,4,5,8]:
        scores_names.append(f"bandwidth={bw}{postfix}")
scores_names = ",".join([str(s) for s in scores_names])
# csv_file.write(
#     f"nclicks,nimages,{scores_names},folder\n")

data = {"postfix": [],
        "nclicks": [],
        "setup": [],
        "bw": [],
        "nimage": [],
        "image_idx": [],
        "SEG": []}
apthreshs = [0.5, 0.75, 0.9]
for th in apthreshs:
    data[f"mAP{int(th*100)}"] = []

for i in tqdm(range(5)):
    expname_plus_setup = f"{expname}_setup_t{i:04}"
    scores = []
    log_file = f"{base_folder}/setup_t{i:04}/output.log"
    file1 = open(log_file, 'r')
    lines = file1.readlines(10000)
    ll = [l for l in lines if "clicks" in l]
    if len(ll) == 0:
        print("Could not find clicks in log", log_file)
        continue

    nclicks = int(ll[0].split(" ")[2])

    scores = []

    nimages = 0
    if f"inference/{expname_plus_setup}/pn_embedding" not in pred_zarr:
        continue

    for postfix in ["", "_full"]:
        for bw in [3,4,5,8]:
            for idx in [1]:#tqdm(pred_zarr[f"inference/{expname_plus_setup}/pn_embedding"], leave=False):
                key = f"inference/{expname_plus_setup}/pn_embedding/{idx}/ms_seg_bw{bw}{postfix}"
                if key in pred_zarr:
                    predicted_segmentation = pred_zarr[key][:]
                    gt_segmentation = gt_zarr[f"{idx}/gt"][:]

                    if postfix=="_full":
                        predicted_segmentation += 1
                        predicted_segmentation[gt_segmentation==0] = 0
                    data["nclicks"].append(nclicks)
                    data["setup"].append(i)
                    data["postfix"].append(postfix)
                    data["bw"].append(bw)
                    data["nimage"].append(1)
                    data["image_idx"].append(idx)
                    data["SEG"].append(seg_metric(predicted_segmentation, gt_segmentation))

                    for th in apthreshs:
                        data[f"mAP{int(th*100)}"].append(matching(gt_segmentation,
                                                                    predicted_segmentation, thresh=th).accuracy)
                else:
                    pass
                    # print("Warning no segmentation found")

df = pd.DataFrame(data)
aggfunc = {'nimage': np.sum, 'SEG': np.mean}
for th in apthreshs:
    aggfunc[f"mAP{int(th*100)}"] = np.mean

table = pd.pivot_table(df, 
                        values=["nimage", "SEG", "mAP50", "mAP75", "mAP90"],
                       index=["setup", "nclicks"],
                       columns=["postfix", 'bw'],
                       aggfunc=aggfunc)
table.to_csv(outfile)



