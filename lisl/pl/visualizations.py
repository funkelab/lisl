from skimage.io import imsave
import io

import matplotlib
matplotlib.use('agg')
import matplotlib.image as mpimg

import matplotlib.pyplot as plt
import numpy as np

from lisl.pl.utils import label2color

def vis_anchor_embedding(embedding, patch_coords, img, grad=None, output_file=None):
    # patch_coords.shape = (num_patches, 2)

    if img.shape[0] not in [3]:
      plt.imshow(img[0], cmap='magma', interpolation='nearest')
    else:
      plt.imshow(np.transpose(img, (1, 2, 0)), interpolation='nearest')

    plt.quiver(patch_coords[:, 0],
               patch_coords[:, 1],
               embedding[:, 0],
               embedding[:, 1], 
               angles='xy',
               scale_units='xy',
               scale=1., color='#8fffdd')

    if grad is not None:
        plt.quiver(patch_coords[:, 0],
                   patch_coords[:, 1],
                   grad[:, 0],
                   grad[:, 1],
                   angles='xy',
                   scale_units='xy',
                   scale=None,
                   color='r')

    plt.axis('off')

    if output_file is not None:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')

    # buf = io.BytesIO()
    # plt.savefig(buf, format='png')
    # buf.seek(0)
    plt.cla()
    plt.clf()
    # return buf