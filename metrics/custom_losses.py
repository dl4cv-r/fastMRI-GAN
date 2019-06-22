from torch import nn
import torch.nn.functional as F

from metrics.my_ssim import ssim_loss


class CSSIM(nn.Module):  # Complementary SSIM
    def __init__(self, default_range=1, filter_size=11, k1=0.01, k2=0.03, sigma=1.5, reduction='mean'):
        super().__init__()
        self.max_val = default_range
        self.filter_size = filter_size
        self.k1 = k1
        self.k2 = k2
        self.sigma = sigma
        self.reduction = reduction

    def forward(self, input, target, max_val=None):
        max_val = self.max_val if max_val is None else max_val
        return 1 - ssim_loss(input, target, max_val=max_val, filter_size=self.filter_size, k1=self.k1, k2=self.k2,
                             sigma=self.sigma, reduction=self.reduction)


class L1CSSIM(nn.Module):  # Replace this with a system of summing losses in Model Trainer later on.
    def __init__(self, l1_weight, default_range=1, filter_size=11, k1=0.01, k2=0.03, sigma=1.5, reduction='mean'):
        super().__init__()
        self.l1_weight = l1_weight
        self.max_val = default_range
        self.filter_size = filter_size
        self.k1 = k1
        self.k2 = k2
        self.sigma = sigma
        self.reduction = reduction

    def forward(self, input, target, max_val=None):
        max_val = self.max_val if max_val is None else max_val

        cssim = 1 - ssim_loss(input, target, max_val=max_val, filter_size=self.filter_size, k1=self.k1, k2=self.k2,
                              sigma=self.sigma, reduction=self.reduction)

        l1_loss = F.l1_loss(input, target, reduction=self.reduction)

        return cssim + self.l1_weight * l1_loss
