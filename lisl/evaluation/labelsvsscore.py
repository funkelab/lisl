from ctcmetrics.seg import seg_metric
import zarr
from tqdm import tqdm
import numpy as np
import pandas as pd
from EmbedSeg.utils.utils2 import matching

expname = "pn_dsb_04"
expname = "pn_dsb_dev16"
expname = "pn_dsb_05"
expname = "pn_dsb_25"

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
    data[f"fp{int(th*100)}"] = []
    data[f"tp{int(th*100)}"] = []
    data[f"fn{int(th*100)}"] = []
    data[f"n_true{int(th*100)}"] = []
    data[f"n_pred{int(th*100)}"] = []

    data[f"precision{int(th*100)}"] = []
    data[f"recall{int(th*100)}"] = []
    
for i in tqdm(range(10)):
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
            for idx in tqdm(pred_zarr[f"inference/{expname_plus_setup}/pn_embedding"], leave=False):
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
                        match = matching(gt_segmentation, predicted_segmentation, thresh=th)
                        data[f"mAP{int(th*100)}"].append(match.accuracy)
                        data[f"fp{int(th*100)}"].append(match.fp)
                        data[f"tp{int(th*100)}"].append(match.tp)
                        data[f"fn{int(th*100)}"].append(match.fn)
                        data[f"n_true{int(th*100)}"].append(match.n_true)
                        data[f"n_pred{int(th*100)}"].append(match.n_pred)

                        data[f"precision{int(th*100)}"].append(match.precision)
                        data[f"recall{int(th*100)}"].append(match.recall)
                        # data[f"mAP{int(th*100)}"].append(match.accuracy)
                else:
                    pass
                    # print("Warning no segmentation found")

df = pd.DataFrame(data)
print([k for k in data.keys()])
aggfunc = {'nimage': np.sum, 'SEG': np.mean}
values = ["nimage", "SEG"]
for th in apthreshs:
    values.append(f"mAP{int(th*100)}")
    aggfunc[f"mAP{int(th*100)}"] = np.mean

    values.append(f"precision{int(th*100)}")
    aggfunc[f"precision{int(th*100)}"] = np.mean
    values.append(f"recall{int(th*100)}")
    aggfunc[f"recall{int(th*100)}"] = np.mean
    
    values.append(f"fp{int(th*100)}")
    aggfunc[f"fp{int(th*100)}"] = np.sum
    values.append(f"tp{int(th*100)}")
    aggfunc[f"tp{int(th*100)}"] = np.sum
    values.append(f"fn{int(th*100)}")
    aggfunc[f"fn{int(th*100)}"] = np.sum
    values.append(f"n_true{int(th*100)}")
    aggfunc[f"n_true{int(th*100)}"] = np.sum
    values.append(f"n_pred{int(th*100)}")
    aggfunc[f"n_pred{int(th*100)}"] = np.sum

table = pd.pivot_table(df, 
                       values=values,
                       index=["setup", "nclicks"],
                       columns=["postfix", 'bw'],
                       aggfunc=aggfunc)
table.to_csv(outfile)



