from glob import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import json
import numpy as np

import seaborn as sns
sns.set_theme(style="darkgrid")

num_setups = 308

experiment_folders = ["/nrs/funke/wolfs2/lisl/experiments/unet_combinations_3/",
                      "/nrs/funke/wolfs2/lisl/experiments/unet_combinations_5/"]

    # read ds limit from train script
def read_training_args(train_script):
    with open(train_script, "r") as ts:
        lines = ts.readlines()
        run_command = lines[-1]
        args = run_command.split(" --")
    return args


all_stats = []

for experiment_folder in experiment_folders:
    for setup_folder in glob(experiment_folder + f"01_train/setup_t*"):

        train_script = setup_folder + "/train.sh"
        train_args = read_training_args(train_script)

        ds_limit = [s.split(" ")[1:] for s in train_args if s.startswith("ds_limit")]
        if ds_limit and len(ds_limit[0]) > 1:
            ds_limit = ds_limit[0]
            ds_limit = int(ds_limit[1]) - int(ds_limit[0])
        else:
            # set maximum dsl dataset length
            ds_limit = 445

        emb_keys = "_".join([s.split(" ")[1:] for s in train_args if s.startswith("emb_keys")][0])

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
                        data_dict["emb_keys"] = emb_keys
                        data_dict["threshold"] = threshold
                        data_dict["stat_key"] = stat_key
                        data_dict["setup"] = setup
                        data_dict["score"] = stats[threshold][stat_key]
                        all_stats.append(data_dict)


df = pd.DataFrame.from_dict(all_stats)

for th in ["0.5", "0.6", "0.8", "0.9"]:
    for score_name in ['precision', 'recall', 'fp', 'tp', 'fn', 'precision', 'recall',
                       'accuracy', 'f1', 'mean_true_score',
                       'mean_matched_score', 'panoptic_quality']:
        ax = None
        for emb_keys in ['raw_',
                        'raw_train/prediction_',
                        'raw_simclr_',
                        'raw_train/prediction_simclr_',
                        'raw_train/prediction_cooc_up1.25_cooc_up1.5_cooc_up1.75_cooc_up2.0_cooc_up3.0_cooc_up4.0_',
                        'raw_train/prediction_cooc_up1.25_cooc_up1.5_cooc_up1.75_cooc_up2.0_cooc_up3.0_cooc_up4.0_simclr_',
                        ]:

            sel_embedkey = df[df["emb_keys"]==emb_keys]
            sel_key_score = sel_embedkey[sel_embedkey["stat_key"]==score_name]
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
            print(emb_keys, subs_grouped)
            ax = sns.lineplot(data=subs_grouped, x="ds_limit", y="score", ax=ax)

            
        ax.set_xlabel("number of annotated images")
        ax.set_ylabel(score_name)
        plt.xscale("log")
        ax.legend(['raw',
                    'raw + cooc',
                    'raw + simclr',
                    'raw + cooc + simclr',
                    'raw + cooc(multiscale)',
                    'raw + cooc(multiscale) + simclr']);
        # ax.set_title(f"{score_name} with threshold {th}")
        ax.figure.savefig(f'plots/{score_name}_{th}.png')
        plt.close("all")

# any score
# 
# over   number of labled images