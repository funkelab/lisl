from glob import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import json
import numpy as np
import os

import seaborn as sns
sns.set_theme(style="darkgrid")

num_setups = 308

exp_name = "resnet_init_2"
experiment_folders = [#"/nrs/funke/wolfs2/lisl/experiments/unet_combinations_3/",
                      #"/nrs/funke/wolfs2/lisl/experiments/unet_combinations_5/"]
                    #   "/nrs/funke/wolfs2/lisl/experiments/unet_combinations_8/",
                      "/nrs/funke/wolfs2/lisl/experiments/resnet_init_2"]

    # read ds limit from train script
def read_training_args(train_script):
    with open(train_script, "r") as ts:
        lines = ts.readlines()
        run_command = lines[-1]
        args = run_command.split(" --")
    return args


all_stats = []

for experiment_folder in experiment_folders:
    for setup_folder in glob(experiment_folder + f"02_train*/setup_t*"):

        train_script = setup_folder + "/train.sh"
        if not os.path.isfile(train_script):
            train_script = setup_folder + "/done.sh"
            
        train_args = read_training_args(train_script)

        ds_limit = [s.split(" ")[1:] for s in train_args if s.startswith("ds_limit")]
        if ds_limit and len(ds_limit[0]) > 1:
            ds_limit = ds_limit[0]
            ds_start = int(ds_limit[0])
            ds_limit = int(ds_limit[1]) - int(ds_limit[0])
        else:
            # set maximum dsl dataset length
            ds_limit = 445

        emb_keys = "_".join([s.split(" ")[1:] for s in train_args if s.startswith("emb_keys")][0])

        if len(glob(setup_folder+"/val_stats_*.json")) == 0:
            print(setup_folder, " val_stats_ not found!")        

        for stats_file in glob(setup_folder+"/val_stats_*.json"):
            iteration = int(stats_file.split("_")[-1][:-5])
            setup = stats_file.split("/")[-2]
            with open(stats_file) as f:
                stats = json.load(f)
                for threshold in stats:
                    for stat_key in stats[threshold]:
                        data_dict = {}
                        data_dict["iteration"] = iteration
                        data_dict["ds_limit"] = ds_limit
                        data_dict["ds_tmp"] = ds_start + (1e8 * ds_limit)
                        data_dict["emb_keys"] = emb_keys
                        data_dict["threshold"] = threshold
                        data_dict["stat_key"] = stat_key
                        data_dict["setup"] = setup
                        data_dict["score"] = stats[threshold][stat_key]
                        all_stats.append(data_dict)


df = pd.DataFrame.from_dict(all_stats)

emb_keys_dict = {
        'raw_': 'raw',
        'raw_train/prediction_': 'raw + cooc',
        'raw_simclr_': 'raw + simclr',
        # 'raw_train/prediction_simclr_': 'raw + cooc + simclr',
        'raw_train/prediction_cooc_up1.25_cooc_up1.5_cooc_up1.75_cooc_up2.0_cooc_up3.0_cooc_up4.0_': 'raw + cooc(multiscale)',
        # 'raw_train/prediction_cooc_up1.25_cooc_up1.5_cooc_up1.75_cooc_up2.0_cooc_up3.0_cooc_up4.0_simclr_': 'raw + cooc(multiscale) + simclr',
}

for th in ["0.5", "0.6", "0.8", "0.9"]:
    for score_name in ['precision', 'recall', 'fp', 'tp', 'fn', 'precision', 'recall',
                       'accuracy', 'f1', 'mean_true_score',
                       'mean_matched_score', 'panoptic_quality']:
        ax = None
        legend = []
        
        refkey = "raw_train/prediction_cooc_up1.25_cooc_up1.5_cooc_up1.75_cooc_up2.0_cooc_up3.0_cooc_up4.0_"
        valid_refs = np.unique(df[df["emb_keys"]==refkey]["ds_tmp"])
        
        for emb_keys in emb_keys_dict:

            sel_embedkey = df[df["emb_keys"]==emb_keys]
            sel_key_score = sel_embedkey[sel_embedkey["stat_key"]==score_name]

            sel_key_score = sel_key_score[sel_embedkey["ds_tmp"].isin(valid_refs)]

            sel_key_score_th = sel_key_score[sel_key_score["threshold"]==th]
            subs = sel_key_score_th[["ds_limit", "setup", "score"]]
            subs = subs.astype({'score': 'float'})
            # pick maximum within each setup
            subs_grouped = subs.groupby(by=['setup', 'ds_limit']).agg({'score': 'max'})
            # compute mean and std of all runs
            # grp = subs_grouped.groupby('ds_limit')
            # means = grp.agg({'score': np.mean})
            # stds = grp.agg({'score': np.std})
            # subs_grouped = subs_grouped[subs_grouped.index != 2]

            # print(emb_keys, subs_grouped)
            # ax = subs_grouped.plot(y='score', ax=ax)
            print(emb_keys, subs_grouped.shape)
            ax = sns.lineplot(data=subs_grouped, x="ds_limit", y="score", ax=ax, legend='brief', label=emb_keys_dict[emb_keys])
            # legend.append()

            
        ax.set_xlabel("number of annotated images")
        ax.set_ylabel(score_name)
        plt.xscale("log")
        # ax.legend();
        # ax.set_title(f"{score_name} with threshold {th}")
        os.makedirs(f'plots/{exp_name}/', exist_ok=True)

        ax.figure.savefig(f'plots/{exp_name}/{score_name}_{th}.png')
        plt.close("all")

# any score
# 
# over   number of labled images