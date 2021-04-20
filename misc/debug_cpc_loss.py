
from lisl.pl.trainer import SSLTrainer
import torch
import torch.optim as optim
from skimage.io import imsave
from torch.nn import functional as F


n_channels = 8
iterations = 200

sslt = SSLTrainer(loss_name="CPC",
                  distance=16,
                  head_layers=0,
                  encoder_layers=0,
                  ndim=2,
                  hidden_channels=n_channels,
                  in_channels=1,
                  out_channels=3,
                  stride=4,
                  initial_lr=1e-4)


embedding = torch.randn((2, n_channels, 16, 16), requires_grad=True, device="cuda")
embedding = F.upsample(embedding, scale_factor=8, mode='bicubic')
# embedding[:, 0, 64:-64, 64:-64] += 0.01
# embedding[:, 1, 64:-64, 64:-64] -= 0.01

embedding = embedding.detach().requires_grad_()

sslt = sslt.cuda()

grads = [None]

# if y_hat.requires_grad:
def log_hook(grad_input):
    print("logging", grad_input.shape)
    grads[0] = grad_input
    # torch.cat((grad_input.detach().cpu(), y_hat.detach().cpu()), dim=0)
    # grad_input_batch = torch.cat(tuple(torch.cat(tuple(vis(e_0[c]) for c in range(e_0.shape[0])), dim=1) for e_0 in grad_input), dim=2)
    # self.logger.experiment.add_image(f'train_regression_grad', grad_input_batch, self.global_step)
    # handle.remove()

handle = embedding.register_hook(log_hook)

optimizer = optim.SGD(list(sslt.parameters()) + [embedding, ], lr=1e1)


for it in range(0, iterations):
    optimizer.zero_grad()
    loss, _, _ = sslt.loss_CPCshift(None, embedding, (4,2))
    loss.backward()
    vis_grad = grads[0].detach()
    vis_grad = vis_grad / (vis_grad.abs().max() + 1e-8)
    vis_grad = vis_grad.detach().cpu().numpy()

    vis_emb = embedding.detach()
    vis_emb = vis_emb / (vis_emb.abs().max() + 1e-8)
    vis_emb = vis_emb.detach().cpu().numpy()

    for c in range(1):
        imsave(f"img/grad_{c:03d}_{it:03d}.png", vis_grad[0, c])
        imsave(f"img/emb_{c:03d}_{it:03d}.png", vis_emb[0, c])

    print(embedding.mean())
    optimizer.step()
