import torch


class Model(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def get_output_shape(self, input_shape):
        return self(torch.rand(*(input_shape))).data.shape
