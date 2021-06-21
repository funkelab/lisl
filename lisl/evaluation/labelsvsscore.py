from ctcmetrics.seg import seg_metric
import zarr
from tqdm import tqdm
import numpy as np
import pandas as pd
from EmbedSeg.utils.utils2 import matching
from skimage.io import imsave
import os
import flow_vis
from skimage import color


expname = "pn_dsb_04"
expname = "pn_dsb_dev16"
# expname = "pn_dsb_25"
expname = "comb_05"
# expname = "suppone_01"
TRANSPOSE = False
SAVEIMG = False

# prediction_file = "/nrs/funke/wolfs2/lisl/datasets/fast_dsb_img_test_out.zarr"
prediction_file = "/nrs/funke/wolfs2/lisl/datasets/fast_dsb_coord_test_inference.zarr"


# expname = "pn_dsb_05"
# prediction_file = "/nrs/funke/wolfs2/lisl/datasets/fast_dsb_img_test.zarr"
# TRANSPOSE = True




gt_file = "/nrs/funke/wolfs2/lisl/datasets/fast_dsb_coord_test2.zarr"

base_folder = f"/nrs/funke/wolfs2/lisl/experiments/{expname}/03_fast/"
pred_zarr = zarr.open(prediction_file, "r")
gt_zarr = zarr.open(gt_file, "r")
outfile = f"/nrs/funke/wolfs2/lisl/experiments/{expname}/labelvsscore_sf.csv"
# outfile = f"/nrs/funke/wolfs2/lisl/experiments/{expname}/labelvsscore_sf.csv"
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
        "sf": [],
        "nimage": [],
        "image_idx": [],
        "SEG": []}
apthreshs = [0.5]#, 0.75, 0.9]
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
        print("pn_embedding not found for ", f"inference/{expname_plus_setup}/pn_embedding" , " in ", prediction_file)
        continue

    for postfix in ["", "_full"]:
        for bw in [12]:
            for sf in [0, 10, 20, 50, 100]:
                for idx in tqdm(pred_zarr[f"inference/{expname_plus_setup}/pn_embedding"], leave=False):

                    # if idx not in ["34", "31"]:
                    #     continue

                    key = f"inference/{expname_plus_setup}/pn_embedding/{idx}/ms_seg_bw{bw}{postfix}"
                    if key in pred_zarr:
                        predicted_segmentation = pred_zarr[key][:]
                        gt_segmentation = gt_zarr[f"{idx}/gt"][:]

                        embedding = pred_zarr[f"inference/{expname_plus_setup}/pn_embedding/{idx}/inst_embedding"][:].transpose(1,2,0)
                        
                        print(embedding.max(), embedding.min())
                        
                        if TRANSPOSE:
                            embedding[..., 0] -= np.arange(embedding.shape[1])[None, :]
                            embedding[..., 1] -= np.arange(embedding.shape[0])[:, None]
                        else:
                            embedding[..., 0] -= np.arange(embedding.shape[0])[:, None]
                            embedding[..., 1] -= np.arange(embedding.shape[1])[None, :]

                        print(embedding.shape)
                        outpath = f"/groups/funke/home/wolfs2/local/data/lsl/evaluation/{expname_plus_setup}"
                        base_name = f"{i:02}_{bw}_{sf}_{idx}_{postfix}"

                        os.makedirs(outpath, exist_ok=True)
                        # imsave(
                        #     f"{outpath}/{base_name}_0_embedding.png", embedding[..., :3])
                        # print(embedding.shape)
                        # exit()

                        if SAVEIMG:
                            rgb = flow_vis.flow_to_color(embedding[..., :2], convert_to_bgr=False)
                            imsave(
                                f"{outpath}/{base_name}_0_embedding.png", rgb)

                        seg = pred_zarr[f"inference/{expname_plus_setup}/pn_embedding/{idx}/inst_embedding"][:].transpose(1,2,0)


                        if postfix=="_full":
                            predicted_segmentation += 1
                            predicted_segmentation[gt_segmentation==0] = 0

                        # size filter
                        if sf > 0:
                            values, counts = np.unique(
                                predicted_segmentation, return_counts=True)
                            # could be vectorized!
                            for filter_idx, c in zip(values, counts):
                                if c < sf:
                                    predicted_segmentation[predicted_segmentation==filter_idx] = 0

                        data["nclicks"].append(nclicks)
                        data["setup"].append(i)
                        data["postfix"].append(postfix)
                        data["bw"].append(bw)
                        data["sf"].append(sf)
                        data["nimage"].append(1)
                        data["image_idx"].append(idx)
                        data["SEG"].append(seg_metric(predicted_segmentation, gt_segmentation))

                        for th in apthreshs:
                            match = matching(gt_segmentation, predicted_segmentation, thresh=th, report_matches=True)

                            # debug image
                            pred_img = np.zeros(gt_segmentation.shape)
                            gt_match_img = np.zeros(gt_segmentation.shape)
                            pred_match_img = np.zeros(gt_segmentation.shape)
                            match_img = np.zeros(gt_segmentation.shape)
                            # match_img[..., 0][predicted_segmentation>0] = 1
                            # match_img[..., 2][gt_segmentation > 0] = 1.

                            for match_gt_idx, mscore in zip(match.matched_pairs, match.matched_scores):
                                print("match", match_gt_idx, mscore)
                                # match_img[..., 2] = (raw - raw.min())/(raw.max()-raw.min())
                                gt_match_img[gt_segmentation == match_gt_idx[0]] = mscore
                                pred_match_img[predicted_segmentation ==
                                               match_gt_idx[1]] = mscore


                            if SAVEIMG:
                                raw = gt_zarr[f"{idx}/raw"][:]
                                imsave(
                                    f"{outpath}/{base_name}_1_pred.png", color.label2rgb(predicted_segmentation))
                                imsave(
                                    f"{outpath}/{base_name}_4_gt.png", color.label2rgb(gt_segmentation))
                                imsave(
                                    f"{outpath}/{base_name}_5_raw.png", raw)
                            # imsave(
                            #     f"{outpath}/{base_name}_3_gtmatch.png", gt_match_img)
                            # imsave(
                            #     f"{outpath}/{base_name}_2_prematch.png", pred_match_img)
                        #    imsave(
                        #         f"{outpath}/{base_name}_false_positives.png", match_img)
                        #    imsave(
                        #         f"{outpath}/{base_name}_false_negatives.png", match_img)


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
print(data)
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
                       columns=["postfix", 'bw', 'sf'],
                       aggfunc=aggfunc)
table.to_csv(outfile)



