from torch import nn


class OutputReconstructionTransform(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, recons, extra_params):
        batch_size = recons.size(0)
        std = extra_params['std'].view(batch_size, 1, 1, 1).to(recons.device)
        mean = extra_params['mean'].view(batch_size, 1, 1, 1).to(recons.device)
        return recons * std + mean
