import torch
from torch import nn, autograd, cuda
from torch.utils.data import DataLoader
import numpy as np
import h5py
from tqdm import tqdm

from pathlib import Path
from collections import defaultdict

from utils.train_utils import load_model_from_checkpoint
from data.mri_data import HDF5Dataset
from models.unet import UnetModel
from utils.arguments import create_arg_parser
from data.pre_processing import InputTestTransform
from data.post_processing import OutputReconstructionTransform


class ModelEvaluator:
    def __init__(self, model, checkpoint_path, pre_processing, post_processing, data_dir, out_dir, device):

        assert isinstance(model, nn.Module)
        assert callable(pre_processing), '`pre_processing` must be a callable function.'
        assert callable(post_processing), 'post_processing must be a callable function.'

        torch.autograd.set_grad_enabled(False)
        self.model = load_model_from_checkpoint(model, checkpoint_path).to(device)
        print(f'Loaded model parameters from {checkpoint_path}')
        self.model.eval()

        self.pre_processing = pre_processing
        self.post_processing = post_processing
        self.data_dir = data_dir
        self.out_dir = Path(out_dir)
        self.device = device

    def _create_reconstructions(self, data_loader):
        reconstructions = defaultdict(list)

        for ds_slices, file_names, s_idxs, extra_params in tqdm(data_loader):
            recons = self.model(ds_slices.to(self.device))
            recons = self.post_processing(recons, extra_params).cpu().numpy()

            for idx in range(recons.shape[0]):
                file_name = Path(file_names[idx]).name
                reconstructions[file_name].append((int(s_idxs[idx]), recons[idx, ...].squeeze()))

        reconstructions = {
            file_name: np.stack([recon for _, recon in sorted(recons_list)])
            for file_name, recons_list in reconstructions.items()
        }

        return reconstructions

    def _create_data_loader(self):
        dataset = HDF5Dataset(root=self.data_dir, transform=self.pre_processing, training=False)
        data_loader = DataLoader(dataset, batch_size=1, num_workers=1, pin_memory=True)
        return data_loader

    def _save_reconstructions(self, reconstructions):
        """
        Saves the reconstructions from a model into h5 files that is appropriate for submission
        to the leaderboard. Also includes compression for faster data transfer.
        Args:
            reconstructions (dict[str, np.array]): A dictionary mapping input filenames to
                corresponding reconstructions (of shape num_slices x height x width).
        """
        self.out_dir.mkdir(exist_ok=False)  # Throw an error to prevent overwriting.
        gzip = dict(compression='gzip', compression_opts=1, shuffle=True, fletcher32=True)
        for file_name, recons in tqdm(reconstructions.items()):
            with h5py.File(self.out_dir / file_name, mode='x', libver='latest') as f:  # Overwriting throws an error.
                f.create_dataset('reconstruction', data=recons, **gzip)

    def create_and_save_reconstructions(self):
        data_loader = self._create_data_loader()
        print('Beginning Reconstruction.')
        reconstructions = self._create_reconstructions(data_loader)
        print('Beginning Saving.')
        self._save_reconstructions(reconstructions)


def main(model, args):
    # Selecting device
    if (args.gpu is not None) and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')

    print(f'Device {device} has been selected.')

    pre_processing = InputTestTransform()
    post_processing = OutputReconstructionTransform()

    evaluator = ModelEvaluator(
        model, args.checkpoint_path, pre_processing, post_processing, args.data_dir, args.out_dir, device)

    evaluator.create_and_save_reconstructions()


if __name__ == '__main__':
    print(f'Current Working Directory: {Path.cwd()}')
    defaults = dict(
        gpu=0,  # Set to None for CPU mode.
        data_dir='/home/veritas/PycharmProjects/fastMRI-GAN/images/multicoil_test',
        checkpoint_path='/home/veritas/PycharmProjects/fastMRI-GAN/checkpoints/'
                        'WGANGP/Trial 07  2019-06-25 11-44-20/Generator/ckpt_001.tar',

        out_dir='./wgan_gp_test'  # Change this every time! Attempted overrides will throw errors by design.
    )

    parser = create_arg_parser(defaults).parse_args()

    # Change this part when a different model is being used.
    unet = UnetModel(in_chans=1, out_chans=1, chans=32, num_pool_layers=4, drop_prob=0)

    main(unet, parser)
