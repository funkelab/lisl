from torch.nn.modules.module import Module
from torch import Tensor

class AnchorLoss(Module):
    r"""

    Args:
        anchor_radius (float): The attraction of anchor points is non linear (sigmoid function).
            The used nonlinear distance is sigmoid(euclidean_distance - anchor_radius)
            Therefore the anchors experience a stronger attraction when their distance is 
            smaller than the anchor_radius
        
    Shape:
        - Input: :math:`(N, C)` where `C = number of classes`, or
          :math:`(N, C, d_1, d_2, ..., d_K)` with :math:`K \geq 1`
          in the case of `K`-dimensional loss.
    """

    def __init__(self, anchor_radius) -> None:
        super().__init__()
        self.anchor_radius = anchor_radius

    def forward(self, embedding, abs_coords, patch_mask=None) -> Tensor:

        abs_embedding = embedding + abs_coords


        coord_diff = abs_coords[:, :, None] - abs_coords[:, None]
        e0 = embedding[:, :, None]
        e1 = embedding[:, None]

        # compute all pairwise distances of anchor embeddings
        diff = (e0 - e1) + coord_diff
        dist = diff.norm(2, dim=-1)
        # dist.shape = (b, p, p)

        nonlinear_dist = (dist - self.anchor_radius).sigmoid()
        
        # only matched patches (e.g. patches close in proximity)
        # contripute to the loss
        if patch_mask is not None:
            nonlinear_dist = nonlinear_dist[patch_mask]

        return nonlinear_dist.sum()