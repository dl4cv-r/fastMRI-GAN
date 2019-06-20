import torch
from torch import nn


class OutputTrainTransform(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, targets, extra_params):
        assert outputs.size() == targets.size(), 'Incorrect shape!'
        batch_size = outputs.size(0)
        std = extra_params['std'].view(batch_size, 1, 1, 1)
        mean = extra_params['mean'].view(batch_size, 1, 1, 1)


