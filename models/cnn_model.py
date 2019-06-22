from torch import nn


class SimpleCNN(nn.Module):
    def __init__(self, chans=32, negative_slope=0.2):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=chans, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.Conv2d(in_channels=chans, out_channels=chans * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=chans * 2),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.Conv2d(in_channels=chans * 2, out_channels=chans*4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=chans * 4),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.Conv2d(in_channels=chans * 4, out_channels=chans * 8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=chans * 8),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.Conv2d(in_channels=chans * 8, out_channels=1, kernel_size=3, padding=1, bias=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Sigmoid()
        )

    def forward(self, tensor):
        return self.layers(tensor)
