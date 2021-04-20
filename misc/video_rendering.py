from glob import glob
import argparse
import subprocess
from natsort import natsorted
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('searchpattern', type=str, help='search pattern for image frames')
    parser.add_argument('loss_file', type=str)

    args = parser.parse_args()
    max_iteration = 2000

    msft = pd.read_csv(args.loss_file)

    print(msft)
    msft.set_index('Step').resample(1)
    print(msft)

    with open("frame_list.txt", "w") as frame_list:
        with open("frame_list_table.txt", "w") as frame_list2:
            for fn in natsorted(glob(args.searchpattern)):
                iteration = int(fn.split("/")[-2])

                print(iteration)

                if iteration < max_iteration:
                    # msft[msft['Step'] < max(iteration, 100)].plot("Step", "Value")
                    sns.lineplot(data=msft[msft['Step'] < max(iteration, 100)], x="Step", y="Value")

                    plt.xlim(right=max_iteration)
                    plt.ylim(bottom=50000)

                    plt.savefig(f"tmp/{iteration:09}.png", dpi=200)
                    plt.close()
                    print(f"iteration {iteration}")

                    frame_list.write(f"file \'{fn}\'\nduration 0.5\n")
                    frame_list2.write(f"file \'tmp/{iteration:09}.png\'\nduration 0.5\n")


    # subprocess.call(['ffmpeg', '-f', 'concat', '-i', 'frame_list.txt', 'out.mp4'])

        