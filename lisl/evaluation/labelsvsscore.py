from ctcmetrics.seg import seg_metric
import zarr
from tqdm import tqdm
import numpy as np

expname = "pn_dsb_04"
expname = "pn_dsb_dev16"
base_folder = f"/nrs/funke/wolfs2/lisl/experiments/{expname}/03_fast/"
prediction_file = "/nrs/funke/wolfs2/lisl/datasets/fast_dsb_img_test.zarr"

pred_zarr = zarr.open(prediction_file, "r")
outfile = f"/nrs/funke/wolfs2/lisl/experiments/{expname}/labelvsscore.csv"

with open(outfile, "w") as csv_file:
    for i in tqdm(range(37)):
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
        for postfix in ["", "_full"]:
            for bw in [3,4,5,8]:
                scores.append([])

                for idx in tqdm(pred_zarr[f"inference/{expname_plus_setup}/pn_embedding"], leave=False):
                    key = f"inference/{expname_plus_setup}/pn_embedding/{idx}/ms_seg_bw{bw}{postfix}"
                    if key in pred_zarr:
                        predicted_segmentation = pred_zarr[key][:]
                        gt_segmentation = pred_zarr[f"gt/{idx}"][:]

                        if postfix=="_full":
                            predicted_segmentation += 1
                            predicted_segmentation[gt_segmentation==0] = 0
                            scores[-1].append(seg_metric(predicted_segmentation, gt_segmentation))
                        else:
                            scores[-1].append(seg_metric(predicted_segmentation, gt_segmentation))
                        nimages += 1
                    else:
                        pass
                        # print("Warning no segmentation found")

            # print(f"{nclicks},{np.mean(scores)}")
        print(scores)
        scores = [np.mean(s) for s in scores]
        scores_string = ",".join([str(s) for s in scores])
        csv_file.write(
            f"{nclicks},{nimages},{scores_string},{base_folder}/setup_t{i:04}\n")
        csv_file.flush()


