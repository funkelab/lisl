from PIL import Image
import torch
import json
import os


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


def is_jsonable(x):
    try:
        json.dumps(x)
        return True
    except BaseException:
        return False


def save_args(args, directory):
    os.mkdir(directory)
    log_out = os.path.join(directory, "commandline_args.txt")
    print(args.__dict__)

    serializable_args = {key: value for (key, value) in args.__dict__.items() if is_jsonable(value)}

    print(serializable_args)

    with open(log_out, 'w') as f:
        json.dump(serializable_args, f, indent=2)

#  gradient vis funtion

# if y_hat.requires_grad:
#     def log_hook(grad_input):
#         # torch.cat((grad_input.detach().cpu(), y_hat.detach().cpu()), dim=0)
#         grad_input_batch = torch.cat(tuple(torch.cat(tuple(vis(e_0[c]) for c in range(e_0.shape[0])), dim=1) for e_0 in grad_input), dim=2)
#         self.logger.experiment.add_image(f'train_regression_grad', grad_input_batch, self.global_step)
#         handle.remove()

#     handle = y_hat.register_hook(log_hook)
