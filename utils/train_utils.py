import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from pathlib import Path

from data.mri_data import HDF5Dataset


class CheckpointManager:
    """
    A checkpoint manager for Pytorch models and optimizers loosely based on Keras/Tensorflow Checkpointers.
    I should note that I am not sure whether this works in Pytorch graph mode.
    Giving up on saving as HDF5 files like in Keras. Just too annoying.
    Note that the whole system is based on 1 indexing, not 0 indexing.
    """
    def __init__(self, model, optimizer, mode='min', save_best_only=True,
                 ckpt_dir='./checkpoints', max_to_keep=5, verbose=True):

        # Type checking.
        assert isinstance(model, nn.Module), 'Not a Pytorch Model'
        assert isinstance(optimizer, optim.Optimizer), 'Not a Pytorch Optimizer'
        assert isinstance(max_to_keep, int) and (max_to_keep >= 0), 'Not a non-negative integer'
        assert mode in ('min', 'max'), 'Mode must be either `min` or `max`'
        ckpt_path = Path(ckpt_dir)
        assert ckpt_path.exists(), 'Not a valid, existing path'

        record_path = ckpt_path / 'Checkpoints.txt'

        try:
            record_file = open(record_path, mode='x')
        except FileExistsError:
            import warnings
            warnings.warn('WARNING: It is recommended to have a separate checkpoint directory for each run.')
            warnings.warn('Appending to previous Checkpoint record file!')
            record_file = open(record_path, mode='a')

        print(f'Checkpoint List for {ckpt_path}', file=record_file)
        record_file.close()

        self.model = model
        self.optimizer = optimizer
        self.save_best_only = save_best_only
        self.ckpt_path = ckpt_path
        self.max_to_keep = max_to_keep
        self.verbose = verbose
        self.save_counter = 0
        self.record_path = record_path
        self.record_dict = dict()

        if mode == 'min':
            self.prev_best = float('inf')
            self.mode = mode
        elif mode == 'max':
            self.prev_best = -float('inf')
            self.mode = mode
        else:
            raise TypeError('Mode must be either `min` or `max`')

    def _save(self, ckpt_name=None, **save_kwargs):
        self.save_counter += 1
        save_dict = {'model_state_dict': self.model.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict()}
        save_dict.update(save_kwargs)
        save_path = self.ckpt_path / (f'{ckpt_name}.tar' if ckpt_name else f'ckpt_{self.save_counter:03d}.tar')

        torch.save(save_dict, save_path)
        if self.verbose:
            print(f'Saved Checkpoint to {save_path}')
            print(f'Checkpoint {self.save_counter:04d}: {save_path}')

        with open(file=self.record_path, mode='a') as file:
            print(f'Checkpoint {self.save_counter:04d}: {save_path}', file=file)

        self.record_dict[self.save_counter] = save_path

        if self.save_counter > self.max_to_keep:
            for count, ckpt_path in self.record_dict.items():  # This system uses 1 indexing.
                if (count <= (self.save_counter - self.max_to_keep)) and ckpt_path.exists():
                    ckpt_path.unlink()  # Delete existing checkpoint

        return save_path

    def save(self, metric, ckpt_name=None, **save_kwargs):  # save_kwargs are extra variables to save
        if self.mode == 'min':
            is_best = metric < self.prev_best
        elif self.mode == 'max':
            is_best = metric > self.prev_best
        else:
            raise TypeError('Mode must be either `min` or `max`')

        save_path = None
        if is_best or not self.save_best_only:
            save_path = self._save(ckpt_name, **save_kwargs)

        if self.verbose:
            if is_best:
                print(f'Metric improved from {self.prev_best:.4e} to {metric:.4e}')
            else:
                print(f'Metric did not improve.')

        if is_best:  # Update new best metric.
            self.prev_best = metric

        # Returns where the file was saved if any was saved. Also returns whether this was the best on the metric.
        return save_path, is_best  # So that one can see whether this one is the best or not.

    def load(self, load_dir, load_optimizer=True):
        save_dict = torch.load(load_dir)
        self.model.load_state_dict(save_dict['model_state_dict'])
        print(f'Loaded model parameters from {load_dir}')

        if load_optimizer:
            self.optimizer.load_state_dict(save_dict['optimizer_state_dict'])
            print(f'Loaded optimizer parameters from {load_dir}')

    def load_latest(self, load_root):
        load_root = Path(load_root)
        load_dir = sorted([x for x in load_root.iterdir() if x.is_dir()])[-1]
        load_file = sorted([x for x in load_dir.iterdir() if x.is_file()])[-1]

        print('Loading', load_file)
        self.load(load_file, load_optimizer=False)
        print('Done')


def load_model_from_checkpoint(model, load_dir):
    """
    A simple function for loading checkpoints without having to use Checkpoint Manager. Very useful for evaluation.
    Checkpoint manager was designed for loading checkpoints before resuming training.

    model (nn.Module): Model architecture to be used.
    load_dir (str): File path to the checkpoint file. Can also be a Path instead of a string.
    """
    assert isinstance(model, nn.Module), 'Model must be a Pytorch module.'
    assert Path(load_dir).exists(), 'The specified directory does not exist'

    save_dict = torch.load(load_dir)
    model.load_state_dict(save_dict['model_state_dict'])
    return model  # Not actually necessary to return the model but doing so anyway.


def create_datasets(args, train_transform, val_transform):

    assert callable(train_transform) and callable(val_transform), 'Transforms should be callable functions.'

    # Generating Datasets.
    train_dataset = HDF5Dataset(
        root=Path(args.data_root) / f'{args.challenge}_train',
        transform=train_transform,
    )

    val_dataset = HDF5Dataset(
        root=Path(args.data_root) / f'{args.challenge}_val',
        transform=val_transform,
    )
    return train_dataset, val_dataset


def create_data_loaders(args, train_transform, val_transform):

    assert callable(train_transform) and callable(val_transform), 'Transforms should be callable functions.'

    train_dataset, val_dataset = create_datasets(args, train_transform, val_transform)

    # Generating Data Loaders
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory
    )

    return train_loader, val_loader


def make_grid_triplet(recons, targets):

    assert recons.size() == targets.size()

    recons = recons.detach().squeeze()
    targets = targets.detach().squeeze()

    small = targets.min()
    large = targets.max()
    diff = large - small

    if recons.dim() == 3:  # batch_size > 1
        recons = torch.cat(torch.split(recons, split_size_or_sections=1, dim=0), dim=1).squeeze()
        targets = torch.cat(torch.split(targets, split_size_or_sections=1, dim=0), dim=1).squeeze()

    recon_grid = (recons - small) / diff
    recon_grid = recon_grid.cpu().numpy()
    target_grid = (targets - small) / diff
    target_grid = target_grid.cpu().numpy()
    delta_grid = target_grid - recon_grid

    return recon_grid, target_grid, delta_grid
