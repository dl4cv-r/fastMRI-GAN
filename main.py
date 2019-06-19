import torch
from torch import cuda, optim

from pathlib import Path

from utils.arguments import create_arg_parser
from utils.run_utils import initialize, get_logger, save_dict_as_json
from utils.train_utils import create_data_loaders
from train.model_trainer import ModelTrainer
from models.unet import UnetModel
from data.pre_processing import InputTrainTransform
from metrics.bad_ssim import CSSIM


def train():
    defaults = dict(
            challenge='multicoil',
            batch_size=1,
            num_workers=1,
            init_lr=1E-3,
            log_dir='./logs',
            ckpt_dir='./checkpoints',
            gpu=0,  # Set to None for CPU mode.
            num_epochs=2,
            max_to_keep=1,
            verbose=False,
            save_best_only=True,
            data_root='./images',
            max_images=6,  # Maximum number of images to save.
            chans=32,
            num_pool_layers=4,
            pin_memory=True,
            add_graph=False
        )

    # Replace with a proper argument parsing function later.
    args = create_arg_parser(defaults).parse_args()

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

    # DataLoaders
    train_loader, val_loader = create_data_loaders(
        args=args, train_transform=InputTrainTransform(), val_transform=InputTrainTransform())

    # Loss Function and output post-processing functions.
    loss_func = CSSIM(window_size=7, val_range=12)

    # Define model.
    model = UnetModel(
        in_chans=1, out_chans=1, chans=args.chans, num_pool_layers=args.num_pool_layers, drop_prob=0).to(device)

    optimizer = optim.Adam(params=model.parameters(), lr=args.init_lr)

    trainer = ModelTrainer(
        args, model=model, optimizer=optimizer, train_loader=train_loader, val_loader=val_loader, loss_func=loss_func)

    trainer.train_model()


if __name__ == '__main__':
    train()
