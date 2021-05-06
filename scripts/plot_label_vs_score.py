import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

from glob import glob
from tqdm import tqdm
import pandas as pd

pattern = f"/nrs/funke/wolfs2/lisl/experiments/pn_dsb_25/labelvsscore.csv"

for fn in glob(pattern):
# for fn in ["/nrs/funke/wolfs2/lisl/experiments/pn_dsb_14/labelvsscore.csv"]:
    data = pd.read_csv(fn, header=[0, 1, 2, 3], skipinitialspace=True)
    print([k for k in data.keys()])
    nclicks_key = [k for k in data.keys() if 'nclicks' in k][0]

        # data = pd.read_csv(fn, names=['nclicks', 'nimages', 'bandwidth=3', 'bandwidth=4', 'bandwidth=5', 'bandwidth=8',
        #                         'bandwidth=3_foreground=gt', 'bandwidth=4_foreground=gt', 'bandwidth=5_foreground=gt', 'bandwidth=8_foreground=gt', 'folder'])
    # if "nclicks"
    for full in ["_full", "Unnamed"]:
        for postfix in ['SEG', 'recall', 'mAP', 'precision', 'fp', 'fn']:
            outfile = f'/nrs/funke/wolfs2/lisl/experiments/tables/' + \
                fn.split('/')[-2] + f"_{postfix}{full}.png"
            f = plt.figure()
            print()
            y = [k for k in data.keys() if any(
                [_.startswith(postfix) for _ in k])]
            print("y", y)
            y = [k for k in y if full in k[1]]
            # print([k for k in data.keys() if any([_==full for _ in k]))
            fig = data.plot(x=nclicks_key, y=y, logx=True, ax=f.gca())
            plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
            fig.figure.savefig(outfile, bbox_inches='tight')

    # outfile = f'/nrs/funke/wolfs2/lisl/experiments/tables/' + fn.split('/')[-2] + ".png"
    # f = plt.figure()
    # fig = data.plot(x=nclicks_key, y=[k for k in data.keys() if any(
    #     [_.startswith('mAP') for _ in k])], logx=True, ax=f.gca())
    # plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    # fig.figure.savefig(outfile, bbox_inches='tight')
