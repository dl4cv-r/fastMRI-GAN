import torch
from torch import nn


class CAModule(nn.Module):
    def __init__(self, num_channels, reduction=16, use_gap=True, use_gmp=True, use_bn1=True, use_bn2=True):
        super().__init__()

        if not (use_gap or use_gmp):
            raise ValueError('Impossible configuration.')

        self.use_gap = use_gap
        self.use_gmp = use_gmp
        self.use_bn1 = use_bn1
        self.use_bn2 = use_bn2

        self.bn1 = nn.BatchNorm1d()
        self.bn2 = nn.BatchNorm1d()
        self.fc1 = nn.Linear(in_features=num_channels, out_features=num_channels // reduction)
        self.fc2 = nn.Linear(in_features=num_channels // reduction, out_features=num_channels)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, tensor):

        # Assuming all conditions are on.

        gap = self.gap(tensor)
        gmp = self.gmp(tensor)

        gap = self.bn1(gap)
        gmp = self.bn1(gmp)

        gap = self.fc1(gap)
        gmp = self.fc1(gmp)

        gap = self.relu(gap)
        gmp = self.relu(gmp)

        gap = self.bn2(gap)
        gmp = self.bn2(gmp)

        gap = self.fc2(gap)
        gmp = self.fc2(gmp)

        output = self.sigmoid(gap + gmp).view(tensor.size(0), tensor.size(1), 1, 1)

        return tensor * output












