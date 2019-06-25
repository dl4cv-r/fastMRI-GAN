import torch
from torch import cuda, optim, nn

from pathlib import Path

from utils.arguments import create_arg_parser
from utils.run_utils import initialize, get_logger, save_dict_as_json
from utils.train_utils import create_data_loaders
from train.model_trainer import ModelTrainer
from train.gan_model_trainer import GANModelTrainer
from train.wgan_gp_model_trainer import ModelTrainerWGANGP
from models.unet import UnetModel
from models.cnn_model import SimpleCNN, SimpleCNNNBN
from data.pre_processing import InputTrainTransform
from metrics.custom_losses import L1CSSIM


def train_supervised_model(args):

    # Creating checkpoint and logging directories, as well as the run name.
    ckpt_path = Path(args.ckpt_dir)
    ckpt_path.mkdir(exist_ok=True)

    run_number, run_name = initialize(ckpt_path)

    ckpt_path = ckpt_path / run_name
    ckpt_path.mkdir(exist_ok=True)

    log_path = Path(args.log_dir)
    log_path.mkdir(exist_ok=True)
    log_path = log_path / run_name
    log_path.mkdir(exist_ok=True)

    logger = get_logger(name=__name__, save_file=log_path / run_name)

    # Assignment inside running code appears to work.
    if (args.gpu is not None) and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
        logger.info(f'Using GPU {args.gpu} for {run_name}')
    else:
        device = torch.device('cpu')
        logger.info(f'Using CPU for {run_name}')

    # Please note that many objects (such as Path objects) cannot be serialized to json files.
    save_dict_as_json(vars(args), log_dir=log_path, save_name=run_name)

    # Saving peripheral variables and objects in args to reduce clutter and make the structure flexible.
    args.run_number = run_number
    args.run_name = run_name
    args.ckpt_path = ckpt_path
    args.log_path = log_path
    args.device = device

    train_transform = InputTrainTransform(is_training=True)
    val_transform = InputTrainTransform(is_training=False)

    # DataLoaders
    train_loader, val_loader = create_data_loaders(args, train_transform, val_transform)

    # Loss Function and output post-processing functions.
    loss_func = L1CSSIM(l1_weight=args.l1_weight, default_range=12, filter_size=7, reduction='mean')

    # Define model.
    model = UnetModel(1, 1, args.chans, args.num_pool_layers, drop_prob=0).to(device)

    optimizer = optim.Adam(params=model.parameters(), lr=args.init_lr)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)

    trainer = ModelTrainer(args, model=model, optimizer=optimizer, train_loader=train_loader, val_loader=val_loader,
                           loss_func=loss_func, scheduler=scheduler)

    trainer.train_model()


def train_gan_model(args):
    # Creating checkpoint and logging directories, as well as the run name.
    ckpt_path = Path(args.ckpt_dir)
    ckpt_path.mkdir(exist_ok=True)

    ckpt_path = ckpt_path / args.train_method
    ckpt_path.mkdir(exist_ok=True)

    run_number, run_name = initialize(ckpt_path)

    ckpt_path = ckpt_path / run_name
    ckpt_path.mkdir(exist_ok=True)

    log_path = Path(args.log_dir)
    log_path.mkdir(exist_ok=True)

    log_path = log_path / args.train_method
    log_path.mkdir(exist_ok=True)

    log_path = log_path / run_name
    log_path.mkdir(exist_ok=True)

    logger = get_logger(name=__name__, save_file=log_path / run_name)

    # Assignment inside running code appears to work.
    if (args.gpu is not None) and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
        logger.info(f'Using GPU {args.gpu} for {run_name}')
    else:
        device = torch.device('cpu')
        logger.info(f'Using CPU for {run_name}')

    # Please note that many objects (such as Path objects) cannot be serialized to json files.
    save_dict_as_json(vars(args), log_dir=log_path, save_name=run_name)

    # Saving peripheral variables and objects in args to reduce clutter and make the structure flexible.
    args.run_number = run_number
    args.run_name = run_name
    args.ckpt_path = ckpt_path
    args.log_path = log_path
    args.device = device

    train_transform = InputTrainTransform(is_training=True)
    val_transform = InputTrainTransform(is_training=False)

    # DataLoaders
    train_loader, val_loader = create_data_loaders(args, train_transform, val_transform)

    # Loss Function and output post-processing functions.
    gan_loss_func = nn.BCELoss(reduction='mean')
    recon_loss_func = L1CSSIM(l1_weight=args.l1_weight, default_range=12, filter_size=7, reduction='mean')
    # recon_loss_func = nn.L1Loss(reduction='mean')
    loss_funcs = {'gan_loss_func': gan_loss_func, 'recon_loss_func': recon_loss_func}

    # Define model.
    generator = UnetModel(1, 1, args.chans, args.num_pool_layers, drop_prob=0).to(device)
    discriminator = SimpleCNN(chans=args.chans).to(device)

    gen_optim = optim.Adam(params=generator.parameters(), lr=args.init_lr)
    disc_optim = optim.Adam(params=discriminator.parameters(), lr=args.init_lr)

    gen_scheduler = optim.lr_scheduler.StepLR(gen_optim, step_size=args.step_size, gamma=args.lr_reduction_rate)
    disc_scheduler = optim.lr_scheduler.StepLR(disc_optim, step_size=args.step_size, gamma=args.lr_reduction_rate)

    trainer = GANModelTrainer(args, generator, discriminator, gen_optim, disc_optim, train_loader, val_loader,
                              loss_funcs, gen_scheduler, disc_scheduler)

    trainer.train_model()


def train_wgan_gp_model(args):
    # Creating checkpoint and logging directories, as well as the run name.
    ckpt_path = Path(args.ckpt_dir)
    ckpt_path.mkdir(exist_ok=True)

    ckpt_path = ckpt_path / args.train_method
    ckpt_path.mkdir(exist_ok=True)

    run_number, run_name = initialize(ckpt_path)

    ckpt_path = ckpt_path / run_name
    ckpt_path.mkdir(exist_ok=True)

    log_path = Path(args.log_dir)
    log_path.mkdir(exist_ok=True)

    log_path = log_path / args.train_method
    log_path.mkdir(exist_ok=True)

    log_path = log_path / run_name
    log_path.mkdir(exist_ok=True)

    logger = get_logger(name=__name__, save_file=log_path / run_name)

    # Assignment inside running code appears to work.
    if (args.gpu is not None) and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
        logger.info(f'Using GPU {args.gpu} for {run_name}')
    else:
        device = torch.device('cpu')
        logger.info(f'Using CPU for {run_name}')

    # Please note that many objects (such as Path objects) cannot be serialized to json files.
    save_dict_as_json(vars(args), log_dir=log_path, save_name=run_name)

    # Saving peripheral variables and objects in args to reduce clutter and make the structure flexible.
    args.run_number = run_number
    args.run_name = run_name
    args.ckpt_path = ckpt_path
    args.log_path = log_path
    args.device = device

    train_transform = InputTrainTransform(is_training=True)
    val_transform = InputTrainTransform(is_training=False)

    # DataLoaders
    train_loader, val_loader = create_data_loaders(args, train_transform, val_transform)

    # Loss Function and output post-processing functions.
    gan_loss_func = nn.BCELoss(reduction='mean')
    recon_loss_func = L1CSSIM(l1_weight=args.l1_weight, default_range=12, filter_size=7, reduction='mean')
    # recon_loss_func = nn.L1Loss(reduction='mean')
    loss_funcs = {'gan_loss_func': gan_loss_func, 'recon_loss_func': recon_loss_func}

    # Define model.
    generator = UnetModel(1, 1, args.chans, args.num_pool_layers, drop_prob=0).to(device)
    # generator = nn.DataParallel(generator)
    discriminator = SimpleCNNNBN(chans=args.chans).to(device)
    # discriminator = nn.DataParallel(discriminator)

    gen_optim = optim.Adam(params=generator.parameters(), lr=args.init_lr)
    disc_optim = optim.Adam(params=discriminator.parameters(), lr=args.init_lr)

    gen_scheduler = optim.lr_scheduler.StepLR(gen_optim, step_size=args.step_size, gamma=args.lr_reduction_rate)
    disc_scheduler = optim.lr_scheduler.StepLR(disc_optim, step_size=args.step_size, gamma=args.lr_reduction_rate)

    trainer = ModelTrainerWGANGP(args, generator, discriminator, gen_optim, disc_optim, train_loader, val_loader,
                                 loss_funcs, gen_scheduler, disc_scheduler)

    trainer.train_model()


if __name__ == '__main__':
    defaults = dict(
        sample_rate=1,
        challenge='multicoil',
        batch_size=1,
        num_workers=1,
        init_lr=1E-4,
        log_dir='./logs',
        ckpt_dir='./checkpoints',
        gpu=1,  # Set to None for CPU mode.
        num_epochs=10,
        max_to_keep=2,
        verbose=False,
        save_best_only=True,
        data_root='./images',
        max_images=8,  # Maximum number of images to save.
        chans=32,
        num_pool_layers=4,
        pin_memory=True,
        add_graph=False,
        l1_weight=1,
        step_size=5,  # For the learning rate scheduler.
        lr_reduction_rate=0.1,
        recon_lambda=10,
        lambda_gp=10,
        train_method='WGANGP',
        gen_prev_model_ckpt='/home/veritas/PycharmProjects/fastMRI-GAN/checkpoints/'
                            'WGANGP/Trial 05  2019-06-24 18-28-17/Generator/ckpt_004.tar',
        disc_prev_model_ckpt='',
        # prev_model_ckpt='',
    )

    # Replace with a proper argument parsing function later.
    parser = create_arg_parser(defaults).parse_args()
    train_wgan_gp_model(parser)
