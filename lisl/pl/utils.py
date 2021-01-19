from PIL import Image
from gunpowder.batch_request import BatchRequest
import torch
import json
import os
import numpy as np
import scipy.sparse as sparse
import gunpowder as gp
import matplotlib
import random
from torch.nn import functional as F
from argparse import ArgumentParser
import inspect

def offset_slice(offset, reverse=False, extra_dims=0):
    def shift(o):
        if o == 0:
            return slice(None)
        elif o > 0:
            return slice(o, None)
        else:
            return slice(0, o)
    if not reverse:
        return (slice(None),) * extra_dims + tuple(shift(int(o)) for o in offset)
    else:
        return (slice(None),) * extra_dims + tuple(shift(-int(o)) for o in offset)


def label2color(label):

    if isinstance(label, Image.Image):
        label = np.array(label)
        if len(label.shape) == 3:
            label = label[..., 0]

    cmap = matplotlib.cm.get_cmap('nipy_spectral')
    shuffle_labels = np.concatenate(
        ([0], np.random.permutation(label.max()) + 1))
    label = shuffle_labels[label]
    return cmap(label / label.max()).transpose(2, 0, 1)

def try_remove(filename):
    try:
        os.remove(filename)
    except OSError:
        pass

def visnorm(x):
    x = x - x.min()
    x = x / x.max()
    return x

def vis(x, normalize=True):
    if isinstance(x, Image.Image):
        x = np.array(x)

    assert(len(x.shape) in [2, 3])

    if len(x.shape) == 2:
        x = x[None]
    else:
        if x.shape[0] not in [1, 3]:
            if x.shape[2] in [1, 3]:
                x = x.transpose(2, 0, 1)
            else:
                raise Exception(
                    "can not visualize array with shape ", x.shape)

    if normalize:
        with torch.no_grad():
            visnorm(x)

    return x

# def vis(x):

#     if isinstance(x, Image.Image):
#         x = np.array(x)

#     assert(len(x.shape) in [2, 3])

#     if len(x.shape) == 2:
#         x = x[None]
#     else:
#         if x.shape[0] not in [1, 3]:
#             if x.shape[2] in [1, 3]:
#                 x = x.transpose(2, 0, 1)
#             else:
#                 raise Exception(
#                     "can not visualize array with shape ", x.shape)

#     return x

def log_img(name, img):
    pl_module.logger.experiment.add_image(name, img, pl_module.global_step)
    try:
        imsave(os.path.join(eval_directory, name+".png"), img.transpose(2, 0, 1))
    except:
        print("can not imsave ", img.shape)

def is_jsonable(x):
    try:
        json.dumps(x)
        return True
    except BaseException:
        return False


def save_args(args, directory):
    os.mkdir(directory)
    log_out = os.path.join(directory, "commandline_args.txt")
    serializable_args = {key: value for (key, value) in args.__dict__.items() if is_jsonable(value)}

    with open(log_out, 'w') as f:
        json.dump(serializable_args, f, indent=2)


def adapted_rand(seg, gt, all_stats=False, ignore_label=True):
    """Compute Adapted Rand error.
    Parameters
    ----------
    seg : np.ndarray
        the segmentation to score, where each value is the label at that point
    gt : np.ndarray, same shape as seg
        the groundtruth to score against, where each value is a label
    all_stats : boolean, optional
        whether to also return precision and recall as a 3-tuple with rand_error
    ignore_label: boolean, optional
        whether to ignore the zero label
    Returns
    -------
    are : float
        The adapted Rand error; equal to $1 - \frac{2pr}{p + r}$,
        where $p$ and $r$ are the precision and recall described below.
    prec : float, optional
        The adapted Rand precision. (Only returned when `all_stats` is ``True``.)
    rec : float, optional
        The adapted Rand recall.  (Only returned when `all_stats` is ``True``.)
    """
    # segA is truth, segB is query
    segA = np.ravel(gt)
    segB = np.ravel(seg)
    n = segA.size

    n_labels_A = int(np.amax(segA)) + 1
    n_labels_B = int(np.amax(segB)) + 1

    ones_data = np.ones(n)

    p_ij = sparse.csr_matrix(
        (ones_data, (segA[:], segB[:])), shape=(n_labels_A, n_labels_B))

    if ignore_label:
        a = p_ij[1:n_labels_A, :]
        b = p_ij[1:n_labels_A, 1:n_labels_B]
        c = p_ij[1:n_labels_A, 0].todense()
    else:
        a = p_ij[:n_labels_A, :]
        b = p_ij[:n_labels_A, 1:n_labels_B]
        c = p_ij[:n_labels_A, 0].todense()
    d = b.multiply(b)

    a_i = np.array(a.sum(1))
    b_i = np.array(b.sum(0))

    sumA = np.sum(a_i * a_i)
    sumB = np.sum(b_i * b_i) + (np.sum(c) / n)
    sumAB = np.sum(d) + (np.sum(c) / n)

    precision = sumAB / sumB
    recall = sumAB / sumA

    fScore = 2.0 * precision * recall / (precision + recall)
    are = 1.0 - fScore

    if all_stats:
        return {"are": are,
                "precision": precision,
                "recall": recall}
    else:
        return are



def offset_from_direction(direction, max_direction=8., distance=10):
    angle = (direction / max_direction)
    angle = 2 * np.pi * angle

    x_offset = int(0.75 * distance * np.sin(angle))
    y_offset = int(0.75 * distance * np.cos(angle))

    x_offset += random.randint(-int(0.15 * distance),
                               +int(0.15 * distance))
    y_offset += random.randint(-int(0.15 * distance),
                               +int(0.15 * distance))

    return x_offset, y_offset

def random_offset(distance=10):
    angle = 2 * np.pi * np.random.uniform()
    distance = np.random.uniform(low=1., high=distance)

    x_offset = int(distance * np.sin(angle))
    y_offset = int(distance * np.cos(angle))

    return x_offset, y_offset

# if y_hat.requires_grad:
#     def log_hook(grad_input):
#         # torch.cat((grad_input.detach().cpu(), y_hat.detach().cpu()), dim=0)
#         grad_input_batch = torch.cat(tuple(torch.cat(tuple(vis(e_0[c]) for c in range(e_0.shape[0])), dim=1) for e_0 in grad_input), dim=2)
#         self.logger.experiment.add_image(f'train_regression_grad', grad_input_batch, self.global_step)
#         handle.remove()

#     handle = y_hat.register_hook(log_hook)

import scipy
import numbers
from skimage.transform import rescale

class UpSample(gp.nodes.BatchFilter):

    def __init__(self, source, factor, target):

        assert isinstance(source, gp.ArrayKey)
        assert isinstance(target, gp.ArrayKey)
        assert (
            isinstance(factor, numbers.Number) or isinstance(factor, tuple)), (
            "Scaling factor should be a number or a tuple of numbers.")

        self.source = source
        self.factor = factor
        self.target = target

    def setup(self):

        spec = self.spec[self.source].copy()
        spec.roi = spec.roi * self.factor
        self.provides(self.target, spec)
        self.enable_autoskip()

    def prepare(self, request):

        deps = gp.BatchRequest()
        sdep = request[self.target]
        sdep.roi = sdep.roi / self.factor
        deps[self.source] = sdep
        return deps

    def process(self, batch, request):
        outputs = gp.Batch()

        # logger.debug("upsampeling %s with %s", self.source, self.factor)

        # resize
        data = batch.arrays[self.source].data
        data = rescale(data, self.factor)

        # create output array
        spec = self.spec[self.target].copy()
        spec.roi = request[self.target].roi
        outputs.arrays[self.target] = gp.Array(data, spec)
        
        return outputs


class AbsolutIntensityAugment(gp.nodes.BatchFilter):

    def __init__(self, array, scale_min, scale_max, shift_min, shift_max):
        self.array = array
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.shift_min = shift_min
        self.shift_max = shift_max

    def setup(self):
        self.enable_autoskip()
        self.updates(self.array, self.spec[self.array])

    def prepare(self, request):
        deps = BatchRequest()
        deps[self.array] = request[self.array].copy()
        return deps

    def process(self, batch, request):

        raw = batch.arrays[self.array]

        raw.data = self.__augment(raw.data,
                np.random.uniform(low=self.scale_min, high=self.scale_max),
                np.random.uniform(low=self.shift_min, high=self.shift_max))

        # clip values, we might have pushed them out of [0,1]
        raw.data[raw.data>1] = 1
        raw.data[raw.data<0] = 0

    def __augment(self, a, scale, shift):

        return a*scale + shift


class Patchify(object):
    """ Adapted from 
    https://github.com/PyTorchLightning/pytorch-lightning-bolts/blob/8a4cf8f61644c28d6df54ccffe3a52d6f5fce5a6/pl_bolts/transforms/self_supervised/ssl_transforms.py#L62
    This implementation adds a dilation parameter
    """


    def __init__(self, patch_size, overlap_size, dilation):
        self.patch_size = patch_size
        self.overlap_size = self.patch_size - overlap_size
        self.dilation = dilation

    def __call__(self, x):
        x = x.unsqueeze(0)
        b, c, h, w = x.size()

        # patch up the images
        # (b, c, h, w) -> (b, c*patch_size, L)
        x = F.unfold(x,
            kernel_size=self.patch_size,
            stride=self.overlap_size,
            dilation=self.dilation)

        # (b, c*patch_size, L) -> (b, nb_patches, width, height)
        x = x.transpose(2, 1).contiguous().view(b, -1, self.patch_size, self.patch_size)

        # reshape to have (b x patches, c, h, w)
        x = x.view(-1, c, self.patch_size, self.patch_size)

        x = x.squeeze(0)

        return x


class BuildFromArgparse(object):
    @classmethod
    def from_argparse_args(cls, args, **kwargs):

        if isinstance(args, ArgumentParser):
            args = cls.parse_argparser(args)
        params = vars(args)

        # we only want to pass in valid DataModule args, the rest may be user specific
        valid_kwargs = inspect.signature(cls.__init__).parameters
        datamodule_kwargs = dict(
            (name, params[name]) for name in valid_kwargs if name in params
        )
        datamodule_kwargs.update(**kwargs)

        return cls(**datamodule_kwargs)