import torch
from torch import nn, optim, autograd
from torch.utils.data import DataLoader

import numpy as np
from tensorboardX import SummaryWriter
from tqdm import tqdm

from time import time
from collections import defaultdict

from utils.train_utils import CheckpointManager, make_grid_triplet
from utils.run_utils import get_logger


def compute_gradient_penalty(discriminator, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    assert real_samples.device == fake_samples.device, 'Devices of data do not match.'
    # Random weight term for interpolation between real and fake samples
    alpha = torch.rand((real_samples.size(0), 1, 1, 1), device=real_samples.device)
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = discriminator(interpolates)
    fake = torch.ones(d_interpolates.size(), dtype=torch.float32, device=real_samples.device)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty  # x10 has not been applied here yet.


class ModelTrainerWGANGP:
    def __init__(self, args, generator, discriminator, gen_optim, disc_optim,
                 train_loader, val_loader, loss_funcs, gen_scheduler=None, disc_scheduler=None):

        self.logger = get_logger(name=__name__, save_file=args.log_path / args.run_name)

        # Checking whether inputs are correct.
        assert isinstance(generator, nn.Module) and isinstance(discriminator, nn.Module), \
            '`generator` and `discriminator` must be Pytorch Modules.'
        assert isinstance(gen_optim, optim.Optimizer) and isinstance(disc_optim, optim.Optimizer), \
            '`gen_optim` and `disc_optim` must be Pytorch Optimizers.'
        assert isinstance(train_loader, DataLoader) and isinstance(val_loader, DataLoader), \
            '`train_loader` and `val_loader` must be Pytorch DataLoader objects.'

        loss_funcs = nn.ModuleDict(loss_funcs)  # Expected to be a dictionary with names and loss functions.

        if gen_scheduler is not None:
            if isinstance(gen_scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.metric_gen_scheduler = True
            elif isinstance(gen_scheduler, optim.lr_scheduler._LRScheduler):
                self.metric_gen_scheduler = False
            else:
                raise TypeError('`gen_scheduler` must be a Pytorch Learning Rate Scheduler.')

        if disc_scheduler is not None:
            if isinstance(disc_scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.metric_disc_scheduler = True
            elif isinstance(disc_scheduler, optim.lr_scheduler._LRScheduler):
                self.metric_disc_scheduler = False
            else:
                raise TypeError('`disc_scheduler` must be a Pytorch Learning Rate Scheduler.')

        self.generator = generator
        self.discriminator = discriminator
        self.gen_optim = gen_optim
        self.disc_optim = disc_optim
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_funcs = loss_funcs
        self.gen_scheduler = gen_scheduler
        self.disc_scheduler = disc_scheduler
        self.device = args.device
        self.verbose = args.verbose
        self.num_epochs = args.num_epochs
        self.writer = SummaryWriter(str(args.log_path))

        self.recon_lambda = torch.tensor(args.recon_lambda, dtype=torch.float32, device=args.device)
        self.lambda_gp = torch.tensor(args.lambda_gp, dtype=torch.float32, device=args.device)

        # This will work best if batch size is 1, as is recommended. I don't know whether this generalizes.
        self.target_real = torch.tensor(1, dtype=torch.float32, device=args.device)
        self.target_fake = torch.tensor(0, dtype=torch.float32, device=args.device)

        # Display interval of 0 means no display of validation images on TensorBoard.
        if args.max_images <= 0:
            self.display_interval = 0
        else:
            self.display_interval = int(len(self.val_loader.dataset) // (args.max_images * args.batch_size))

        self.generator_checkpoint_manager = CheckpointManager(
            model=self.generator, optimizer=self.gen_optim, mode='min', save_best_only=args.save_best_only,
            ckpt_dir=args.ckpt_path / 'Generator', max_to_keep=args.max_to_keep)

        self.discriminator_checkpoint_manager = CheckpointManager(
            model=self.discriminator, optimizer=self.disc_optim, mode='min', save_best_only=args.save_best_only,
            ckpt_dir=args.ckpt_path / 'Discriminator', max_to_keep=args.max_to_keep)

        # loading from checkpoint if specified.
        if vars(args).get('gen_prev_model_ckpt'):
            self.generator_checkpoint_manager.load(load_dir=args.gen_prev_model_ckpt, load_optimizer=False)

        if vars(args).get('disc_prev_model_ckpt'):
            self.discriminator_checkpoint_manager.load(load_dir=args.disc_prev_model_ckpt, load_optimizer=False)

    def train_model(self):
        self.logger.info('Beginning Training Loop.')
        tic_tic = time()
        for epoch in range(1, self.num_epochs + 1):  # 1 based indexing
            # Training
            tic = time()
            train_epoch_loss, train_epoch_loss_components = self._train_epoch(epoch=epoch)
            toc = int(time() - tic)
            self._log_epoch_outputs(epoch=epoch, epoch_loss=train_epoch_loss,
                                    epoch_loss_components=train_epoch_loss_components, elapsed_secs=toc, training=True)

            # Validation
            tic = time()
            val_epoch_loss, val_epoch_loss_components = self._val_epoch(epoch=epoch)
            toc = int(time() - tic)
            self._log_epoch_outputs(epoch=epoch, epoch_loss=val_epoch_loss,
                                    epoch_loss_components=val_epoch_loss_components, elapsed_secs=toc, training=False)

            self.generator_checkpoint_manager.save(metric=val_epoch_loss, verbose=True)

            if self.gen_scheduler is not None:
                if self.metric_gen_scheduler:  # If the scheduler is a metric based scheduler, include metrics.
                    self.gen_scheduler.step(metrics=val_epoch_loss)
                else:
                    self.gen_scheduler.step()

            if self.disc_scheduler is not None:
                if self.metric_disc_scheduler:
                    self.disc_scheduler.step(metrics=val_epoch_loss)
                else:
                    self.disc_scheduler.step()

        # Finishing Training Loop
        self.writer.close()  # Flushes remaining data to TensorBoard.
        toc_toc = int(time() - tic_tic)
        self.logger.info(f'Finishing Training Loop. Total elapsed time: '
                         f'{toc_toc // 3600} hr {(toc_toc // 60) % 60} min {toc_toc % 60} sec.')

    def _train_step(self, inputs, targets, extra_params):

        assert inputs.size() == targets.size(), 'input and target sizes do not match'

        inputs = inputs.to(self.device)
        targets = targets.to(self.device, non_blocking=True)

        # Train discriminator
        self.disc_optim.zero_grad()
        recons = self.generator(inputs)
        pred_fake = self.discriminator(recons.detach())  # Generated fake image going through discriminator.
        pred_real = self.discriminator(targets)  # Real image going through discriminator.
        gradient_penalty = compute_gradient_penalty(self.discriminator, targets, recons.detach())
        disc_loss = pred_fake.mean() - pred_real.mean() + self.lambda_gp * gradient_penalty
        disc_loss.backward()
        self.disc_optim.step()

        # Train Generator
        self.gen_optim.zero_grad()
        pred_fake = self.discriminator(recons)
        gen_loss = -pred_fake.mean()
        recon_loss = self.loss_funcs['recon_loss_func'](recons, targets)
        total_gen_loss = gen_loss + self.recon_lambda * recon_loss
        total_gen_loss.backward()
        self.gen_optim.step()

        # Just using reconstruction loss since it is the most meaningful.
        step_loss = recon_loss
        step_loss_components = {'pred_fake': pred_fake.mean(), 'pred_real': pred_real.mean(),
                                'gradient_penalty': gradient_penalty, 'disc_loss': disc_loss, 'gen_loss': gen_loss,
                                'recon_loss': recon_loss, 'total_gen_loss': total_gen_loss}

        return step_loss, step_loss_components

    def _train_epoch(self, epoch):
        self.generator.train()
        self.discriminator.train()
        torch.autograd.set_grad_enabled(True)

        epoch_loss = list()  # Appending values to list due to numerical underflow.
        epoch_loss_components = defaultdict(list)

        # labels are fully sampled coil-wise images, not rss or esc.
        train_len = len(self.train_loader.dataset)  # Adding progress bar for convenience.
        for step, (inputs, targets, extra_params) in tqdm(enumerate(self.train_loader, start=1), total=train_len):
            step_loss, step_loss_components = self._train_step(inputs, targets, extra_params)

            # Perhaps not elegant, but NaN values make this necessary.
            epoch_loss.append(step_loss.detach())
            for key, value in step_loss_components.items():
                epoch_loss_components[key].append(value.detach())

            if self.verbose:
                self._log_step_outputs(epoch, step, step_loss, step_loss_components, training=True)

        # Return as scalar value and dict respectively. Remove the inner lists.
        return self._get_epoch_outputs(epoch, epoch_loss, epoch_loss_components, training=True)

    def _val_step(self, inputs, targets, extra_params):
        """
        All extra parameters are to be placed in extra_params.
        This makes the system more flexible.
        """

        inputs = inputs.to(self.device)
        targets = targets.to(self.device, non_blocking=True)

        recons = self.generator(inputs)

        # Discriminator part.
        # pred_fake = self.discriminator(recons)  # Generated fake image going through discriminator.
        # pred_real = self.discriminator(targets)  # Real image going through discriminator.
        # gradient_penalty = compute_gradient_penalty(self.discriminator, targets, recons)
        # disc_loss = pred_fake.mean() - pred_real.mean() + self.lambda_gp * gradient_penalty

        # Generator part.
        # gen_loss = -pred_fake.mean()
        recon_loss = self.loss_funcs['recon_loss_func'](recons, targets)
        # total_gen_loss = gen_loss + self.recon_lambda * recon_loss

        step_loss = recon_loss
        step_loss_components = {'recon_loss': recon_loss}

        return recons, step_loss, step_loss_components

    def _val_epoch(self, epoch):
        self.generator.eval()
        self.discriminator.eval()
        torch.autograd.set_grad_enabled(False)

        epoch_loss = list()  # Appending values to list due to numerical underflow.
        epoch_loss_components = defaultdict(list)

        val_len = len(self.val_loader.dataset)
        for step, (inputs, targets, extra_params) in tqdm(enumerate(self.val_loader, start=1), total=val_len):
            recons, step_loss, step_loss_components = self._val_step(inputs, targets, extra_params)

            # Append to list to prevent errors from NaN and Inf values.
            epoch_loss.append(step_loss)
            for key, value in step_loss_components.items():
                epoch_loss_components[key].append(value)

            if self.verbose:
                self._log_step_outputs(epoch, step, step_loss, step_loss_components, training=False)

            # Save images to TensorBoard.
            # Condition ensures that self.display_interval != 0 and that the step is right for display.
            if self.display_interval and (step % self.display_interval == 0):
                recon_grid, target_grid, delta_grid = make_grid_triplet(recons, targets)

                self.writer.add_image(f'Recons/{step}', recon_grid, global_step=epoch, dataformats='HW')
                self.writer.add_image(f'Targets/{step}', target_grid, global_step=epoch, dataformats='HW')
                self.writer.add_image(f'Deltas/{step}', delta_grid, global_step=epoch, dataformats='HW')

        return self._get_epoch_outputs(epoch, epoch_loss, epoch_loss_components, training=False)

    def _get_epoch_outputs(self, epoch, epoch_loss, epoch_loss_components, training=True):
        mode = 'Training' if training else 'Validation'
        num_slices = len(self.train_loader.dataset) if training else len(self.val_loader.dataset)

        # Checking for nan values.
        epoch_loss = torch.stack(epoch_loss)
        is_finite = torch.isfinite(epoch_loss)
        num_nans = (is_finite.size(0) - is_finite.sum()).item()
        if num_nans > 0:
            self.logger.warning(f'Epoch {epoch} {mode}: {num_nans} NaN values present in {num_slices} slices')
            epoch_loss = torch.mean(epoch_loss[is_finite]).item()
        else:
            epoch_loss = torch.mean(epoch_loss).item()

        for key, value in epoch_loss_components.items():
            epoch_loss_component = torch.stack(value)
            is_finite = torch.isfinite(epoch_loss_component)
            num_nans = (is_finite.size(0) - is_finite.sum()).item()

            if num_nans > 0:
                self.logger.warning(f'Epoch {epoch} {mode} {key}: {num_nans} NaN values present in {num_slices} slices')
                epoch_loss_components[key] = torch.mean(epoch_loss_component[is_finite]).item()
            else:
                epoch_loss_components[key] = torch.mean(epoch_loss_component).item()

        return epoch_loss, epoch_loss_components

    def _log_step_outputs(self, epoch, step, step_loss, step_loss_components, training=True):
        mode = 'Training' if training else 'Validation'
        self.logger.info(f'Epoch {epoch:03d} Step {step:03d} {mode} loss: {step_loss.item():.4e}')
        for key, value in step_loss_components.items():
            self.logger.info(f'Epoch {epoch:03d} Step {step:03d}: {mode} {key}: {value.item():.4e}')

    def _log_epoch_outputs(self, epoch, epoch_loss, epoch_loss_components, elapsed_secs, training=True):
        mode = 'Training' if training else 'Validation'
        self.logger.info(f'Epoch {epoch:03d} {mode}. loss: {epoch_loss:.4e}, '
                         f'Time: {elapsed_secs // 60} min {elapsed_secs % 60} sec')
        self.writer.add_scalar(f'{mode}_epoch_loss', scalar_value=epoch_loss, global_step=epoch)

        for key, value in epoch_loss_components.items():
            self.logger.info(f'Epoch {epoch:03d} {mode}. {key}: {value}')
            self.writer.add_scalar(f'{mode}_epoch_{key}', scalar_value=value, global_step=epoch)


