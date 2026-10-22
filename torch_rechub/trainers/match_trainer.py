import math
import os
import warnings

import torch
import tqdm
from sklearn.metrics import roc_auc_score

from ..basic.callback import EarlyStopper
from ..basic.loss_func import BPRLoss, RegularizationLoss
from ..utils.match import gather_inbatch_logits, inbatch_negative_sampling


class MatchTrainer(object):
    """A general trainer for Matching/Retrieval

    Args:
        model (nn.Module): any matching model with user_tower/item_tower methods.
        mode (int, optional): the training mode, `{0:point-wise, 1:pair-wise, 2:list-wise}`. Defaults to 0.
        optimizer_fn (torch.optim): optimizer function of pytorch (default = `torch.optim.Adam`).
        optimizer_params (dict): parameters of optimizer_fn.
        scheduler_fn (torch.optim.lr_scheduler) : torch scheduling class, eg. `torch.optim.lr_scheduler.StepLR`.
        scheduler_params (dict): parameters of optimizer scheduler_fn.
        n_epoch (int): epoch number of training.
        earlystop_patience (int): how long to wait after last time validation auc improved (default=10).
        device (str): `"cpu"` or `"cuda:0"`.
        gpus (list): legacy GPU ID list for training without in-batch negatives.
        model_path (str): the path you want to save the model (default="./"). Note only save the best weight in the validation data.
        in_batch_neg (bool): train on positive-only pairs on a single device using other items in the batch
            as negatives. The trainer temporarily uses the model's mode='in_batch'.
            With mode=0 or 2 and default sampling options, use full-batch cross entropy
            on tower embeddings using the model's own normalization, scaled by
            model.temperature (default=1.0).
            Paired-data labels are placeholders; session next-item labels provide target IDs.
            Validation AUC still requires binary labels.
            Other rows are treated as negatives, including repeated item IDs.
        in_batch_neg_ratio (int): number of negatives to draw from the batch per positive sample when in_batch_neg is True.
        hard_negative (bool): whether to choose hardest negatives within batch (top-k by score) instead of uniform random.
        sampler_seed (int): optional random seed for in-batch sampler to ease reproducibility/testing.
    """

    def __init__(
        self,
        model,
        mode=0,
        in_batch_neg=False,
        in_batch_neg_ratio=None,
        hard_negative=False,
        sampler_seed=None,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=None,
        regularization_params=None,
        scheduler_fn=None,
        scheduler_params=None,
        n_epoch=10,
        earlystop_patience=10,
        device="cpu",
        gpus=None,
        model_path="./",
        model_logger=None,
    ):
        self.model = model
        if gpus is None:
            gpus = []
        if in_batch_neg and (len(gpus) > 1 or isinstance(model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel))):
            raise ValueError("In-batch training supports a single device. Pass an unwrapped model and use device='cpu' or device='cuda:0'.")
        self._full_inbatch = in_batch_neg and mode in (0, 2) and in_batch_neg_ratio is None and not hard_negative
        self._temperature = getattr(model, 'temperature', 1.0)
        if in_batch_neg:
            if not math.isfinite(self._temperature) or self._temperature <= 0:
                raise ValueError("model.temperature must be finite and greater than zero for in-batch training.")
        if in_batch_neg and (not hasattr(model, 'user_tower') or not hasattr(model, 'item_tower')):
            raise ValueError("In-batch training requires a matching model with user_tower and item_tower methods.")
        self.gpus = gpus
        if len(gpus) > 1:
            print('parallel running on these gpus:', gpus)
            self.model = torch.nn.DataParallel(self.model, device_ids=gpus)
        self.device = torch.device(device)
        self.model.to(self.device)
        self.in_batch_neg = in_batch_neg
        self.in_batch_neg_ratio = in_batch_neg_ratio
        self.hard_negative = hard_negative
        self._sampler_generator = None
        if sampler_seed is not None:
            self._sampler_generator = torch.Generator(device=self.device)
            self._sampler_generator.manual_seed(sampler_seed)
        if optimizer_params is None:
            optimizer_params = {"lr": 1e-3, "weight_decay": 1e-5}
        if regularization_params is None:
            regularization_params = {"embedding_l1": 0.0, "embedding_l2": 0.0, "dense_l1": 0.0, "dense_l2": 0.0}
        self.mode = mode
        if mode == 0:  # point-wise loss, binary cross_entropy
            # With in-batch negatives we treat it as list-wise classification over sampled negatives
            self.criterion = torch.nn.CrossEntropyLoss() if in_batch_neg else torch.nn.BCELoss()
        elif mode == 1:  # pair-wise loss
            self.criterion = BPRLoss()
        elif mode == 2:  # list-wise loss, softmax
            self.criterion = torch.nn.CrossEntropyLoss()
        else:
            raise ValueError("mode only contain value in %s, but got %s" % ([0, 1, 2], mode))
        self.optimizer = optimizer_fn(self.model.parameters(), **optimizer_params)  # default optimizer
        self.scheduler = None
        if scheduler_fn is not None:
            self.scheduler = scheduler_fn(self.optimizer, **scheduler_params)
        self.evaluate_fn = roc_auc_score  # default evaluate function
        self.n_epoch = n_epoch
        self.early_stopper = EarlyStopper(patience=earlystop_patience)
        self.model_path = model_path
        # Initialize regularization loss
        self.reg_loss_fn = RegularizationLoss(**regularization_params)
        self.model_logger = model_logger

    def train_one_epoch(self, data_loader, log_interval=10):
        self.model.train()
        total_loss = 0
        epoch_loss = 0
        batch_count = 0
        skipped_batches = 0
        tk0 = tqdm.tqdm(data_loader, desc="train", smoothing=0, mininterval=1.0)
        for i, (x_dict, y) in enumerate(tk0):
            if self.in_batch_neg and y.size(0) < 2:
                # Skip before running the towers: a singleton has no negatives
                # and may also fail in training-mode BatchNorm.
                skipped_batches += 1
                continue
            x_dict = {k: v.to(self.device) for k, v in x_dict.items()}  # tensor to GPU
            y = y.to(self.device)
            if self.in_batch_neg:
                # Session models can use their existing next-item labels as targets.
                x_dict.setdefault('target', y.long())
                loss = self._inbatch_loss(x_dict)
            else:
                y = y.float() if self.mode == 0 else y.long()
                if self.mode == 1:  # pair_wise
                    pos_score, neg_score = self.model(x_dict)
                    if pos_score.shape == neg_score.shape:
                        pos_score, neg_score = pos_score.reshape(-1), neg_score.reshape(-1)
                    loss = self.criterion(pos_score, neg_score)
                else:
                    y_pred = self.model(x_dict)
                    loss = self.criterion(y_pred, y)

            # Add regularization loss
            reg_loss = self.reg_loss_fn(self.model)
            loss = loss + reg_loss

            # used for debug
            # if i == 0:
            #     print()
            #     if self.mode == 0:
            #         print('pred: ', [f'{float(each):5.2g}' for each in y_pred.detach().cpu().tolist()])
            #         print('truth:', [f'{float(each):5.2g}' for each in y.detach().cpu().tolist()])
            #     elif self.mode == 2:
            #         pred = y_pred.detach().cpu().mean(0)
            #         pred = torch.softmax(pred, dim=0).tolist()
            #         print('pred: ', [f'{float(each):4.2g}' for each in pred])
            #     elif self.mode == 1:
            #         print('pos:', [f'{float(each):5.2g}' for each in pos_score.detach().cpu().tolist()])
            #         print('neg: ', [f'{float(each):5.2g}' for each in neg_score.detach().cpu().tolist()])

            self.model.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            epoch_loss += loss.item()
            batch_count += 1
            if batch_count % log_interval == 0:
                tk0.set_postfix(loss=total_loss / log_interval)
                total_loss = 0

        if self.in_batch_neg and batch_count == 0:
            raise ValueError("No usable in-batch training batches. Use batch_size >= 2 and provide at least two positive pairs.")
        if skipped_batches:
            warnings.warn(f"Skipped {skipped_batches} in-batch training batch(es) with fewer than two samples.", RuntimeWarning, stacklevel=2)
        # Return average epoch loss
        return epoch_loss / batch_count if batch_count > 0 else 0

    def fit(self, train_dataloader, val_dataloader=None):
        model_file = os.path.join(self.model_path, "model.pth")
        for logger in self._iter_loggers():
            logger.log_hyperparams({'n_epoch': self.n_epoch, 'learning_rate': self.optimizer.param_groups[0]['lr'], 'loss_mode': self.mode})

        for epoch_i in range(self.n_epoch):
            print('epoch:', epoch_i)
            train_loss = self.train_one_epoch(train_dataloader)

            for logger in self._iter_loggers():
                logger.log_metrics({'train/loss': train_loss, 'learning_rate': self.optimizer.param_groups[0]['lr']}, step=epoch_i)

            if self.scheduler is not None:
                if epoch_i % self.scheduler.step_size == 0:
                    print("Current lr : {}".format(self.optimizer.state_dict()['param_groups'][0]['lr']))
                self.scheduler.step()  # update lr in epoch level by scheduler

            if val_dataloader:
                auc = self.evaluate(self.model, val_dataloader)
                print('epoch:', epoch_i, 'validation: auc:', auc)

                best_score = self.early_stopper.best_auc
                stop_training = self.early_stopper.stop_training(auc, self.model.state_dict())
                if self.early_stopper.best_auc != best_score:
                    # Keep the previous checkpoint intact if saving is interrupted.
                    torch.save(self.early_stopper.best_weights, model_file + ".tmp")
                    os.replace(model_file + ".tmp", model_file)

                for logger in self._iter_loggers():
                    logger.log_metrics({'val/auc': auc}, step=epoch_i)

                if stop_training:
                    print(f'validation: best auc: {self.early_stopper.best_auc}')
                    break

        if val_dataloader is not None and self.early_stopper.best_weights is not None:
            self.model.load_state_dict(self.early_stopper.best_weights)
        else:
            torch.save(self.model.state_dict(), model_file)  # no validation: save the final model

        for logger in self._iter_loggers():
            logger.finish()

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

    def evaluate(self, model, data_loader):
        model.eval()
        targets, predicts = list(), list()
        with torch.no_grad():
            tk0 = tqdm.tqdm(data_loader, desc="validation", smoothing=0, mininterval=1.0)
            for i, (x_dict, y) in enumerate(tk0):
                x_dict = {k: v.to(self.device) for k, v in x_dict.items()}
                y = y.to(self.device)
                y_pred = model(x_dict)
                targets.extend(y.tolist())
                predicts.extend(y_pred.tolist())
        return self.evaluate_fn(targets, predicts)

    def predict(self, model, data_loader):
        model.eval()
        predicts = list()
        with torch.no_grad():
            tk0 = tqdm.tqdm(data_loader, desc="predict", smoothing=0, mininterval=1.0)
            for i, (x_dict, y) in enumerate(tk0):
                x_dict = {k: v.to(self.device) for k, v in x_dict.items()}
                y = y.to(self.device)
                y_pred = model(x_dict)
                predicts.extend(y_pred.tolist())
        return predicts

    def inference_embedding(self, model, mode, data_loader, model_path):
        # inference
        assert mode in ["user", "item"], "Invalid mode={}.".format(mode)
        original_mode = getattr(model, 'mode', None)
        state = torch.load(os.path.join(model_path, "model.pth"), map_location=self.device, weights_only=True)
        model.load_state_dict(state)
        model = model.to(self.device)
        model.eval()
        predicts = []
        # Temporarily select a tower; later calls should keep the original scoring mode.
        model.mode = mode
        try:
            with torch.no_grad():
                tk0 = tqdm.tqdm(data_loader, desc="%s inference" % (mode), smoothing=0, mininterval=1.0)
                for i, x_dict in enumerate(tk0):
                    x_dict = {k: v.to(self.device) for k, v in x_dict.items()}
                    y_pred = model(x_dict)
                    predicts.append(y_pred.data)
        finally:
            # Restore on both success and failure; exceptions still propagate to the caller.
            model.mode = original_mode
        return torch.cat(predicts, dim=0)

    def export_onnx(self, output_path, mode=None, dummy_input=None, batch_size=2, seq_length=10, opset_version=14, dynamic_batch=True, device=None, verbose=False, onnx_export_kwargs=None):
        """Export the trained matching model to ONNX format.

        This method exports matching/retrieval models (e.g., DSSM, YoutubeDNN, MIND)
        to ONNX format. For dual-tower models, you can export user tower and item
        tower separately for efficient online serving.

        Args:
            output_path (str): Path to save the ONNX model file.
            mode (str, optional): Export mode for dual-tower models:
                - "user": Export only the user tower (for user embedding inference)
                - "item": Export only the item tower (for item embedding inference)
                - None: Export the full model's original scoring output (default).
            dummy_input (dict, optional): Example input dict {feature_name: tensor}.
                If not provided, dummy inputs will be generated automatically.
            batch_size (int): Batch size for auto-generated dummy input (default: 2).
            seq_length (int): Sequence length for SequenceFeature (default: 10).
            opset_version (int): ONNX opset version (default: 14).
            dynamic_batch (bool): Enable dynamic batch size (default: True).
            device (str, optional): Device for export ('cpu', 'cuda', etc.).
                If None, defaults to 'cpu' for maximum compatibility.
            verbose (bool): Print export details (default: False).
            onnx_export_kwargs (dict, optional): Extra kwargs forwarded to ``torch.onnx.export``.
                The exporter tries the dynamo path first and falls back to the
                legacy exporter automatically. Pass ``{\"dynamo\": False}`` here
                to force legacy export when needed.

        Returns:
            bool: True if export succeeded, False otherwise.

        Example:
            >>> trainer = MatchTrainer(dssm_model, mode=0, ...)
            >>> trainer.fit(train_dl)

            >>> # Export user tower for user embedding inference
            >>> trainer.export_onnx("user_tower.onnx", mode="user")

            >>> # Export item tower for item embedding inference
            >>> trainer.export_onnx("item_tower.onnx", mode="item")

            >>> # Export full model (for online similarity computation)
            >>> trainer.export_onnx("full_model.onnx")

            >>> # Export on specific device
            >>> trainer.export_onnx("user_tower.onnx", mode="user", device="cpu")
        """
        from ..utils.onnx_export import ONNXExporter

        # Handle DataParallel wrapped model
        model = self.model.module if hasattr(self.model, 'module') else self.model

        # Store original mode
        original_mode = getattr(model, 'mode', None)

        # Use provided device or default to 'cpu'
        export_device = device if device is not None else 'cpu'

        try:
            exporter = ONNXExporter(model, device=export_device)
            return exporter.export(
                output_path=output_path,
                mode=mode,
                dummy_input=dummy_input,
                batch_size=batch_size,
                seq_length=seq_length,
                opset_version=opset_version,
                dynamic_batch=dynamic_batch,
                verbose=verbose,
                onnx_export_kwargs=onnx_export_kwargs,
            )
        finally:
            # Restore original mode
            if hasattr(model, 'mode'):
                model.mode = original_mode

    def visualization(self, input_data=None, batch_size=2, seq_length=10, depth=3, show_shapes=True, expand_nested=True, save_path=None, graph_name="model", device=None, dpi=300, **kwargs):
        """Draw the model; see utils.visualization.visualize_model for options.

        Inputs are generated from model features unless input_data is provided.
        Example: trainer.visualization(save_path="model.png", depth=3).
        """
        from ..utils.visualization import visualize_model

        return visualize_model(
            self.model,
            input_data=input_data,
            batch_size=batch_size,
            seq_length=seq_length,
            depth=depth,
            show_shapes=show_shapes,
            expand_nested=expand_nested,
            save_path=save_path,
            graph_name=graph_name,
            device=device,
            dpi=dpi,
            **kwargs
        )

    def _inbatch_loss(self, x_dict):
        """Get tower embeddings, choose batch negatives, and apply the configured loss."""
        model = self.model
        original_mode = model.mode
        model.mode = 'in_batch'
        try:
            users, items = model(x_dict)
        finally:
            # Only this forward needs embeddings; restore the usual scoring mode even on failure.
            model.mode = original_mode
        if users is None or items is None:
            raise ValueError("In-batch training needs one positive item per row. Provide item_feature or a target item ID.")
        # Existing towers return either [B, D] or [B, 1, D].
        if users.dim() == 3 and users.size(1) == 1:
            users = users.squeeze(1)
        if items.dim() == 3 and items.size(1) == 1:
            items = items.squeeze(1)
        if users.dim() != 2 or users.shape != items.shape:
            raise ValueError("In-batch user and positive item embeddings must have the same [batch_size, embedding_dim] shape.")

        scores = users @ items.t()
        if hasattr(model, 'correct_inbatch_logits'):
            scores = model.correct_inbatch_logits(scores, x_dict)
        logits = scores / self._temperature
        if self._full_inbatch:
            # Each row's positive item is on the diagonal of the full [B, B] matrix.
            targets = torch.arange(logits.size(0), device=logits.device)
        else:
            neg_indices = inbatch_negative_sampling(scores, neg_ratio=self.in_batch_neg_ratio, hard_negative=self.hard_negative, generator=self._sampler_generator)
            logits = gather_inbatch_logits(logits, neg_indices)
            if self.mode == 1:
                return self.criterion(logits[:, 0], logits[:, 1:], in_batch_neg=True)
            # After gathering, the positive item is in column 0 for every row.
            targets = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
        return self.criterion(logits, targets)
