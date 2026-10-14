import os
import warnings

import torch
import tqdm
from sklearn.metrics import roc_auc_score

from ..basic.callback import EarlyStopper
from ..basic.loss_func import RegularizationLoss


class CTRTrainer(object):
    """A general trainer for single task learning.

    Args:
        model (nn.Module): any single-task ranking model.
        optimizer_fn (torch.optim): optimizer function of pytorch (default = `torch.optim.Adam`).
        optimizer_params (dict): parameters of optimizer_fn.
        scheduler_fn (torch.optim.lr_scheduler) : torch scheduling class, eg. `torch.optim.lr_scheduler.StepLR`.
        scheduler_params (dict): parameters of optimizer scheduler_fn.
        n_epoch (int): epoch number of training.
        earlystop_patience (int): how long to wait after last time validation auc improved (default=10).
        device (str): `"cpu"` or `"cuda:0"`
        gpus (list): id of multi gpu (default=[]). If the length >=1, then the model will wrapped by nn.DataParallel.
        loss_mode (bool): whether the model returns only prediction or prediction with extra loss
            (`True`: `model(x_dict) -> y_pred`, `False`: `model(x_dict) -> (y_pred, other_loss)`).
        model_path (str): the path you want to save the model (default="./"). Note only save the best weight in the validation data.
        embedding_l1 (float): L1 regularization coefficient for embedding parameters (default=0.0).
        embedding_l2 (float): L2 regularization coefficient for embedding parameters (default=0.0).
        dense_l1 (float): L1 regularization coefficient for dense parameters (default=0.0).
        dense_l2 (float): L2 regularization coefficient for dense parameters (default=0.0).
        in_batch_neg (bool): construct negative user-item pairs from positive rows in the batch.
            Defaults to False, preserving the original supervised BCE training.
        item_features (list): item column names or Feature objects to swap together.
            User, history and context columns remain attached to their original row.
        in_batch_neg_ratio (int): negatives per positive, default=1. None uses all other rows.
        sampler_seed (int): optional seed for reproducible in-batch sampling.
    """

    def __init__(
        self,
        model,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=None,
        regularization_params=None,
        scheduler_fn=None,
        scheduler_params=None,
        n_epoch=10,
        earlystop_patience=10,
        device="cpu",
        gpus=None,
        loss_mode=True,
        model_path="./",
        model_logger=None,
        in_batch_neg=False,
        item_features=None,
        in_batch_neg_ratio=1,
        sampler_seed=None,
    ):
        self.in_batch_neg = in_batch_neg
        if isinstance(item_features, str):
            item_features = [item_features]
        self.item_features = [getattr(feature, 'name', feature) for feature in item_features or []]
        if in_batch_neg:
            if not self.item_features or any(not isinstance(name, str) or not name for name in self.item_features):
                raise ValueError("Provide item_features, e.g. ['item_id', 'cate_id'], for in-batch ranking.")
            if len(set(self.item_features)) != len(self.item_features):
                raise ValueError("item_features must not contain duplicate column names.")
            if in_batch_neg_ratio is not None and (not isinstance(in_batch_neg_ratio, int) or isinstance(in_batch_neg_ratio, bool) or in_batch_neg_ratio < 1):
                raise ValueError("in_batch_neg_ratio must be a positive integer or None.")
        self.in_batch_neg_ratio = in_batch_neg_ratio
        self.model = model  # for uniform weights save method in one gpu or multi gpu
        if gpus is None:
            gpus = []
        self.gpus = gpus
        if len(gpus) > 1:
            print('parallel running on these gpus:', gpus)
            self.model = torch.nn.DataParallel(self.model, device_ids=gpus)
        # torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)
        self.model.to(self.device)
        self._sampler_generator = None
        if in_batch_neg and sampler_seed is not None:
            self._sampler_generator = torch.Generator(device=self.device).manual_seed(sampler_seed)
        if optimizer_params is None:
            optimizer_params = {"lr": 1e-3, "weight_decay": 1e-5}
        self.optimizer = optimizer_fn(self.model.parameters(), **optimizer_params)  # default optimizer
        if regularization_params is None:
            regularization_params = {"embedding_l1": 0.0, "embedding_l2": 0.0, "dense_l1": 0.0, "dense_l2": 0.0}
        self.scheduler = None
        if scheduler_fn is not None:
            self.scheduler = scheduler_fn(self.optimizer, **scheduler_params)
        self.loss_mode = loss_mode
        self.criterion = torch.nn.BCELoss()  # default loss cross_entropy
        self.evaluate_fn = roc_auc_score  # default evaluate function
        self.n_epoch = n_epoch
        self.early_stopper = EarlyStopper(patience=earlystop_patience)
        self.model_path = model_path
        # Initialize regularization loss
        self.reg_loss_fn = RegularizationLoss(**regularization_params)
        self.model_logger = model_logger

    def _inbatch_data(self, x_dict, y):
        """Keep positive rows and pair each with other rows' complete item features."""
        y = y.reshape(-1)
        if not ((y == 0) | (y == 1)).all():
            raise ValueError("In-batch ranking requires binary labels; y=1 identifies positive interactions.")
        missing = set(self.item_features) - x_dict.keys()
        if missing:
            raise ValueError(f"Missing item feature columns: {sorted(missing)}")
        positive = y == 1
        x_dict = {name: values[positive] for name, values in x_dict.items()}
        batch_size = int(positive.sum())
        if batch_size < 2:
            return None
        rows = torch.arange(batch_size, device=y.device)
        if self.in_batch_neg_ratio is None:
            offsets = torch.arange(1, batch_size, device=y.device).expand(batch_size, -1)
        else:
            offsets = torch.randint(1, batch_size, (batch_size, self.in_batch_neg_ratio), device=y.device, generator=self._sampler_generator)
        donors = (rows.unsqueeze(1) + offsets) % batch_size
        donors = torch.cat([rows.unsqueeze(1), donors], dim=1)
        owners = rows.repeat_interleave(donors.size(1))
        item_rows = donors.reshape(-1)
        paired = {name: values[item_rows if name in self.item_features else owners] for name, values in x_dict.items()}
        labels = torch.zeros_like(donors, dtype=torch.float32)
        labels[:, 0] = 1
        return paired, labels.reshape(-1)

    def train_one_epoch(self, data_loader, log_interval=10):
        self.model.train()
        total_loss = 0
        epoch_loss = 0
        batch_count = 0
        skipped_batches = 0
        tk0 = tqdm.tqdm(data_loader, desc="train", smoothing=0, mininterval=1.0)
        for i, (x_dict, y) in enumerate(tk0):
            x_dict = {k: v.to(self.device) for k, v in x_dict.items()}  # tensor to GPU
            y = y.to(self.device).float()
            if self.in_batch_neg:
                paired = self._inbatch_data(x_dict, y)
                if paired is None:
                    skipped_batches += 1
                    continue
                x_dict, y = paired
            if self.loss_mode:
                y_pred = self.model(x_dict)
                loss = self.criterion(y_pred, y)
            else:
                y_pred, other_loss = self.model(x_dict)
                if self.in_batch_neg and isinstance(other_loss, torch.Tensor):
                    other_loss = other_loss.mean()
                loss = self.criterion(y_pred, y) + other_loss

            # Add regularization loss
            reg_loss = self.reg_loss_fn(self.model)
            loss = loss + reg_loss

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
            raise ValueError("No usable in-batch ranking batches. Provide at least two positive rows per batch.")
        if skipped_batches:
            warnings.warn(f"Skipped {skipped_batches} batch(es) with fewer than two positive interactions.", RuntimeWarning, stacklevel=2)
        # Return average epoch loss
        return epoch_loss / batch_count if batch_count > 0 else 0

    def fit(self, train_dataloader, val_dataloader=None):
        for logger in self._iter_loggers():
            logger.log_hyperparams({'n_epoch': self.n_epoch, 'learning_rate': self.optimizer.param_groups[0]['lr'], 'loss_mode': self.loss_mode})

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

                for logger in self._iter_loggers():
                    logger.log_metrics({'val/auc': auc}, step=epoch_i)

                if self.early_stopper.stop_training(auc, self.model.state_dict()):
                    print(f'validation: best auc: {self.early_stopper.best_auc}')
                    break

        if val_dataloader is not None and self.early_stopper.best_weights is not None:
            self.model.load_state_dict(self.early_stopper.best_weights)
        torch.save(self.model.state_dict(), os.path.join(self.model_path, "model.pth"))  # save best auc model

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
                # 确保y是float类型且维度为[batch_size, 1]
                y = y.to(self.device).float().view(-1, 1)
                if self.loss_mode:
                    y_pred = model(x_dict)
                else:
                    y_pred, _ = model(x_dict)
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
                if self.loss_mode:
                    y_pred = model(x_dict)
                else:
                    y_pred, _ = model(x_dict)
                predicts.extend(y_pred.tolist())
        return predicts

    def export_onnx(self, output_path, dummy_input=None, batch_size=2, seq_length=10, opset_version=14, dynamic_batch=True, device=None, verbose=False, onnx_export_kwargs=None):
        """Export the trained model to ONNX format.

        This method exports the ranking model (e.g., DeepFM, WideDeep, DCN) to ONNX format
        for deployment. The export is non-invasive and does not modify the model code.

        Args:
            output_path (str): Path to save the ONNX model file.
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
            >>> trainer = CTRTrainer(model, ...)
            >>> trainer.fit(train_dl, val_dl)
            >>> trainer.export_onnx("deepfm.onnx")

            >>> # With custom dummy input
            >>> dummy = {"user_id": torch.tensor([1, 2]), "item_id": torch.tensor([10, 20])}
            >>> trainer.export_onnx("model.onnx", dummy_input=dummy)

            >>> # Export on specific device
            >>> trainer.export_onnx("model.onnx", device="cpu")
        """
        from ..utils.onnx_export import ONNXExporter

        # Handle DataParallel wrapped model
        model = self.model.module if hasattr(self.model, 'module') else self.model

        # Use provided device or default to 'cpu'
        export_device = device if device is not None else 'cpu'

        exporter = ONNXExporter(model, device=export_device)
        return exporter.export(
            output_path=output_path,
            dummy_input=dummy_input,
            batch_size=batch_size,
            seq_length=seq_length,
            opset_version=opset_version,
            dynamic_batch=dynamic_batch,
            verbose=verbose,
            onnx_export_kwargs=onnx_export_kwargs,
        )

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
