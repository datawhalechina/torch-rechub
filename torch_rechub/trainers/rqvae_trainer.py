import os
from time import time

import numpy as np
import torch
from tqdm import tqdm


class Trainer(object):
    """Training utility class for PyTorch models.

    Handles the full training loop including optimization, evaluation,
    checkpointing, and logging.

    Parameters
    ----------
    model : torch.nn.Module
        Model to be trained.
    optimizer_fn : callable, default=torch.optim.Adam
        Optimizer constructor.
    optimizer_params : dict, optional
        Parameters passed to the optimizer.
    scheduler_fn : callable, optional
        Learning rate scheduler constructor.
    scheduler_params : dict, optional
        Parameters passed to the scheduler.
    n_epoch : int, default=10
        Number of training epochs.
    device : str, default='cpu'
        Device used for training.
    model_path : str, default='./'
        Directory to save model checkpoints.
    model_logger : object or list, optional
        Logger instance(s) used for recording metrics.
    eval_step : int, default=50
        Evaluation interval measured in epochs.

    Attributes
    ----------
    best_loss : float
        Best training loss observed so far.
    best_collision_rate : float
        Best collision rate observed during evaluation.
    """

    def __init__(
        self,
        model,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=None,
        scheduler_fn=None,
        scheduler_params=None,
        n_epoch=10,
        device='cpu',
        model_path='./',
        model_logger=None,
        eval_step=50,
    ):
        self.model = model
        self.n_epoch = n_epoch
        self.device = torch.device(device)
        self.model.to(self.device)
        if optimizer_params is None:
            optimizer_params = {"lr": 1e-3, "weight_decay": 1e-5}
        self.optimizer = optimizer_fn(self.model.parameters(), **optimizer_params)  # default optimizer
        self.scheduler = None
        if scheduler_fn is not None:
            self.scheduler = scheduler_fn(self.optimizer, **scheduler_params)
        self.model_path = model_path
        self.model_logger = model_logger
        self.eval_step = eval_step

        self.best_save_heap = []
        self.newest_save_queue = []
        self.best_loss = np.inf
        self.best_collision_rate = np.inf
        self.best_loss_ckpt = "best_loss_model.pth"
        self.best_collision_ckpt = "best_collision_model.pth"

    def _check_nan(self, loss):
        """Check whether the loss value is NaN."""
        if torch.isnan(loss):
            raise ValueError("Training loss is nan")

    def _iter_loggers(self):
        """Return logger instances as a list.

        Returns
        -------
        list
            Active logger instances. Empty when ``model_logger`` is ``None``.
        """
        if self.model_logger is None:
            return []
        if isinstance(self.model_logger, (list, tuple)):
            return list(self.model_logger)
        return [self.model_logger]

    def train_one_epoch(self, data_loader):
        """Train the model for a single epoch.

        Parameters
        ----------
        data_loader : torch.utils.data.DataLoader
            DataLoader providing training batches.

        Returns
        -------
        total_loss : float
            Sum of total training loss over the epoch.
        total_recon_loss : float
            Sum of reconstruction loss over the epoch.
        """
        self.model.train()
        total_loss = 0
        total_recon_loss = 0
        iter_data = tqdm(
            data_loader,
            total=len(data_loader),
            ncols=100,
            desc="train",
        )

        for batch_idx, data in enumerate(iter_data):
            data = data.to(self.device)
            self.optimizer.zero_grad()
            out, rq_loss, indices = self.model(data)
            loss, loss_recon = self.model.compute_loss(out, rq_loss, xs=data)
            self._check_nan(loss)
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            total_loss += loss.item()
            total_recon_loss += loss_recon.item()

        return total_loss, total_recon_loss

    @torch.no_grad()
    def evaluate(self, data_loader):
        """Evaluate the model by computing collision rate.

        Parameters
        ----------
        data_loader : torch.utils.data.DataLoader
            DataLoader providing evaluation data.

        Returns
        -------
        collision_rate : float
            Ratio of duplicate semantic codes among all samples.
        """
        self.model.eval()
        iter_data = tqdm(
            data_loader,
            total=len(data_loader),
            ncols=100,
            desc="evaluating",
        )

        indices_set = set()
        num_sample = 0
        for batch_idx, data in enumerate(iter_data):
            num_sample += len(data)
            data = data.to(self.device)
            indices = self.model.get_indices(data)
            indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
            for index in indices:
                code = "-".join([str(int(_)) for _ in index])
                indices_set.add(code)

        collision_rate = (num_sample - len(list(indices_set))) / num_sample

        return collision_rate

    def fit(self, train_dataloader):
        """Run the full training procedure.

        Performs iterative training, periodic evaluation, metric logging,
        and checkpoint saving.

        Parameters
        ----------
        train_dataloader : torch.utils.data.DataLoader
            DataLoader providing training data.

        Returns
        -------
        best_loss : float
            Best training loss achieved.
        best_collision_rate : float
            Best collision rate achieved during evaluation.
        """

        cur_eval_step = 0

        for logger in self._iter_loggers():
            logger.log_hyperparams({'n_epoch': self.n_epoch, 'learning_rate': self.optimizer.param_groups[0]['lr']})

        for epoch_idx in range(self.n_epoch):
            logs = {}
            # train
            training_start_time = time()
            train_loss, train_recon_loss = self.train_one_epoch(train_dataloader)
            training_end_time = time()

            logs['train/loss'] = train_loss
            logs['train/recon_loss'] = train_recon_loss
            logs['train/epoch_time'] = training_end_time - training_start_time

            # eval
            if (epoch_idx + 1) % self.eval_step == 0:
                valid_start_time = time()
                collision_rate = self.evaluate(train_dataloader)
                valid_end_time = time()
                logs['val/collision_rate'] = collision_rate
                logs['val/epoch_time'] = valid_end_time - valid_start_time

                if train_loss < self.best_loss:
                    self.best_loss = train_loss
                    torch.save(self.model.state_dict(), os.path.join(self.model_path, "model_best_loss.pth"))  # save best model
                    logs['best/train_loss'] = self.best_loss
                if collision_rate < self.best_collision_rate:
                    self.best_collision_rate = collision_rate
                    cur_eval_step = 0
                    torch.save(self.model.state_dict(), os.path.join(self.model_path, "model_best_collision_rate.pth"))  # save best model
                    logs['best/collision_rate'] = self.best_collision_rate
                else:
                    cur_eval_step += 1
                # ========== Log once per epoch ==========
            for logger in self._iter_loggers():
                logger.log_metrics(logs, step=epoch_idx)

        for logger in self._iter_loggers():
            logger.finish()

        return self.best_loss, self.best_collision_rate

    def export_onnx(self, output_path, batch_size=2, opset_version=14, dynamic_batch=True, device=None, verbose=False, onnx_export_kwargs=None):
        """
        Export the trained RQVAE model to ONNX format, including reconstructed output and codebook indices.

        Parameters
        ----------
        output_path : str
            Path to save the ONNX model.
        batch_size : int, optional
            Batch size for the dummy input used in export.
        opset_version : int, optional
            ONNX opset version.
        dynamic_batch : bool, optional
            Whether to enable dynamic batch size.
        device : torch.device or str, optional
            Device to run the export (cpu or cuda). Default: model device.
        verbose : bool, optional
            Whether to print ONNX export debug info.
        onnx_export_kwargs : dict, optional
            Additional kwargs for torch.onnx.export.

        Returns
        -------
        bool
            True if export succeeded, False otherwise.

        Example
        -------
        >>> model = RQVAEModel(in_dim=768, num_emb_list=[64,64], e_dim=64)
        >>> model.train()  # assume model has been trained
        >>> output_path = "rqevae.onnx"
        >>> success = model.export_onnx(output_path, batch_size=4, opset_version=14)
        >>> print(success)
        True

        >>> # Export on specific device
        >>> success = model.export_onnx("rqevae_cpu.onnx", batch_size=4, device="cpu")
        >>> print(success)
        True
        """
        try:
            if device is None:
                device = next(self.parameters()).device
            self.to(device)
            self.eval()

            # Dummy input
            dummy_input = torch.randn(batch_size, self.in_dim, device=device)

            # Dynamic axes
            dynamic_axes_dict = None
            if dynamic_batch:
                dynamic_axes_dict = {'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}, 'indices': {0: 'batch_size'}}

            # Export kwargs
            export_kwargs = dict(
                model=self,
                args=(dummy_input,
                      ),
                f=output_path,
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output',
                              'indices'],
                dynamic_axes=dynamic_axes_dict,
                verbose=verbose,
            )
            if onnx_export_kwargs:
                export_kwargs.update(onnx_export_kwargs)

            # Wrapper forward
            def forward_for_export(x):
                out, _, indices = self.forward(x)
                return out, indices

            torch.onnx.export(forward_for_export, dummy_input, output_path, **export_kwargs)
            print(f"ONNX model with output and indices exported to {output_path}")
            return True
        except Exception as e:
            print(f"Failed to export ONNX model: {e}")
            return False
