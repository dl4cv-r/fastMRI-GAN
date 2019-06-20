import torch
from torch import nn, optim, cuda

from sr.models.srgan import GeneratorResNet, Discriminator, VGG19FeatureExtractor54


def train_models(gpu=0, lr=0.001, betas=(0.9, 0.999)):

    generator = GeneratorResNet(in_channels=1, out_channels=1, n_residual_blocks=16).train()
    discriminator = Discriminator(input_shape=(1, 320, 320)).train()
    feature_extractor = VGG19FeatureExtractor54().eval()

    gan_loss = nn.MSELoss()
    content_loss = nn.L1Loss()

    if gpu is not None and torch.cuda.is_available():
        device = torch.cuda.device(f'cuda:{gpu}')
    else:
        device = torch.cuda.device('cpu')

    generator = generator.to(device)
    discriminator = discriminator.to(device)
    feature_extractor = feature_extractor.to(device)

    gen_optim = optim.Adam(generator.parameters(), lr=lr, betas=betas)
    disc_optim = optim.Adam(generator.parameters(), lr=lr, betas=betas)



