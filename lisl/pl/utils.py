from PIL import Image
import torch
import json
import os
import numpy as np
import scipy.sparse as sparse
import matplotlib

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
            x = x - x.min()
            x = x / x.max()

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


#  gradient vis funtion

# if y_hat.requires_grad:
#     def log_hook(grad_input):
#         # torch.cat((grad_input.detach().cpu(), y_hat.detach().cpu()), dim=0)
#         grad_input_batch = torch.cat(tuple(torch.cat(tuple(vis(e_0[c]) for c in range(e_0.shape[0])), dim=1) for e_0 in grad_input), dim=2)
#         self.logger.experiment.add_image(f'train_regression_grad', grad_input_batch, self.global_step)
#         handle.remove()

#     handle = y_hat.register_hook(log_hook)