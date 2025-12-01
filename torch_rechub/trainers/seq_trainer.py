"""Sequence Generation Model Trainer."""

import os

import torch
import torch.nn as nn
import tqdm

from ..basic.callback import EarlyStopper
from ..basic.loss_func import NCELoss


class SeqTrainer(object):
    """序列生成模型训练器.

    用于训练HSTU等序列生成模型。
    支持CrossEntropyLoss损失函数和生成式评估指标。

    Args:
        model (nn.Module): 要训练的模型
        optimizer_fn (torch.optim): 优化器函数，默认为torch.optim.Adam
        optimizer_params (dict): 优化器参数
        scheduler_fn (torch.optim.lr_scheduler): torch调度器类
        scheduler_params (dict): 调度器参数
        n_epoch (int): 训练轮数，默认10
        earlystop_patience (int): 早停耐心值，默认10
        device (str): 设备，'cpu'或'cuda'，默认'cpu'
        gpus (list): 多GPU的id列表，默认为[]
        model_path (str): 模型保存路径，默认为'./'

    Methods:
        fit: 训练模型
        evaluate: 评估模型
        predict: 生成预测

    Example:
        >>> trainer = SeqTrainer(
        ...     model=model,
        ...     optimizer_fn=torch.optim.Adam,
        ...     optimizer_params={'lr': 1e-3, 'weight_decay': 1e-5},
        ...     device='cuda'
        ... )
        >>> trainer.fit(
        ...     train_loader=train_loader,
        ...     val_loader=val_loader
        ... )
    """

    def __init__(self, model, optimizer_fn=torch.optim.Adam, optimizer_params=None, scheduler_fn=None, scheduler_params=None, n_epoch=10, earlystop_patience=10, device='cpu', gpus=None, model_path='./', loss_type='cross_entropy', loss_params=None):
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
        if optimizer_params is None:
            optimizer_params = {"lr": 1e-3, "weight_decay": 1e-5}
        self.optimizer = optimizer_fn(self.model.parameters(), **optimizer_params)  # default optimizer
        self.scheduler = None
        if scheduler_fn is not None:
            self.scheduler = scheduler_fn(self.optimizer, **scheduler_params)

        # 损失函数
        if loss_type == 'nce':
            if loss_params is None:
                loss_params = {"temperature": 0.1, "ignore_index": 0}
            self.loss_fn = NCELoss(**loss_params)
        else:  # default to cross_entropy
            if loss_params is None:
                loss_params = {"ignore_index": 0}
            self.loss_fn = nn.CrossEntropyLoss(**loss_params)

        self.n_epoch = n_epoch
        self.early_stopper = EarlyStopper(patience=earlystop_patience)
        self.model_path = model_path

    def fit(self, train_dataloader, val_dataloader=None):
        """训练模型.

        Args:
            train_dataloader (DataLoader): 训练数据加载器
            val_dataloader (DataLoader): 验证数据加载器

        Returns:
            dict: 训练历史
        """
        history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}

        for epoch_i in range(self.n_epoch):
            print('epoch:', epoch_i)
            # 训练阶段
            self.train_one_epoch(train_dataloader)
            if self.scheduler is not None:
                if epoch_i % self.scheduler.step_size == 0:
                    print("Current lr : {}".format(self.optimizer.state_dict()['param_groups'][0]['lr']))
                self.scheduler.step()  # update lr in epoch level by scheduler

            # 验证阶段
            if val_dataloader:
                val_loss, val_accuracy = self.evaluate(val_dataloader)
                history['val_loss'].append(val_loss)
                history['val_accuracy'].append(val_accuracy)

                print(f"epoch: {epoch_i}, validation: loss: {val_loss:.4f}, accuracy: {val_accuracy:.4f}")

                # 早停
                if self.early_stopper.stop_training(val_accuracy, self.model.state_dict()):
                    print(f'validation: best accuracy: {self.early_stopper.best_auc}')
                    self.model.load_state_dict(self.early_stopper.best_weights)
                    break

        torch.save(self.model.state_dict(), os.path.join(self.model_path, "model.pth"))  # save best model
        return history

    def train_one_epoch(self, data_loader, log_interval=10):
        """Train the model for a single epoch.

        Args:
            data_loader (DataLoader): Training data loader.
            log_interval (int): Interval (in steps) for logging average loss.

        Returns:
            float: Average training loss for this epoch.
        """
        self.model.train()
        total_loss = 0
        tk0 = tqdm.tqdm(data_loader, desc="train", smoothing=0, mininterval=1.0)
        for i, (seq_tokens, seq_positions, seq_time_diffs, targets) in enumerate(tk0):
            # Move tensors to the target device
            seq_tokens = seq_tokens.to(self.device)
            seq_positions = seq_positions.to(self.device)
            seq_time_diffs = seq_time_diffs.to(self.device)
            targets = targets.to(self.device).squeeze(-1)

            # Forward pass
            logits = self.model(seq_tokens, seq_time_diffs)  # (B, L, V)

            # Compute loss
            # For next-item prediction we only use the last position in the sequence
            # logits[:, -1, :] selects the prediction at the last step for each sequence
            last_logits = logits[:, -1, :]  # (B, V)

            loss = self.loss_fn(last_logits, targets)

            # 反向传播
            self.model.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            if (i + 1) % log_interval == 0:
                tk0.set_postfix(loss=total_loss / log_interval)
                total_loss = 0

    def evaluate(self, data_loader):
        """Evaluate the model on a validation/test data loader.

        Args:
            data_loader (DataLoader): Validation or test data loader.

        Returns:
            tuple: ``(avg_loss, top1_accuracy)``.
        """
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for seq_tokens, seq_positions, seq_time_diffs, targets in tqdm.tqdm(data_loader, desc="evaluating", smoothing=0, mininterval=1.0):
                # Move tensors to the target device
                seq_tokens = seq_tokens.to(self.device)
                seq_positions = seq_positions.to(self.device)
                seq_time_diffs = seq_time_diffs.to(self.device)
                targets = targets.to(self.device).squeeze(-1)

                # Forward pass
                logits = self.model(seq_tokens, seq_time_diffs)  # (B, L, V)

                # Compute loss using only the last position (next-item prediction)
                last_logits = logits[:, -1, :]  # (B, V)

                loss = self.loss_fn(last_logits, targets)
                total_loss += loss.item()

                # Compute top-1 accuracy
                predictions = torch.argmax(last_logits, dim=-1)  # (B,)
                correct = (predictions == targets).sum().item()
                total_correct += correct
                total_samples += targets.numel()

        avg_loss = total_loss / len(data_loader)
        accuracy = total_correct / total_samples

        return avg_loss, accuracy

    def export_onnx(self, output_path, batch_size=2, seq_length=50, vocab_size=None, opset_version=14, dynamic_batch=True, device=None, verbose=False):
        """Export the trained sequence generation model to ONNX format.

        This method exports sequence generation models (e.g., HSTU) to ONNX format.
        Unlike other trainers, sequence models use positional arguments (seq_tokens, seq_time_diffs)
        instead of dict input, making ONNX export more straightforward.

        Args:
            output_path (str): Path to save the ONNX model file.
            batch_size (int): Batch size for dummy input (default: 2).
            seq_length (int): Sequence length for dummy input (default: 50).
            vocab_size (int, optional): Vocabulary size for generating dummy tokens.
                If None, will try to get from model.vocab_size.
            opset_version (int): ONNX opset version (default: 14).
            dynamic_batch (bool): Enable dynamic batch size (default: True).
            device (str, optional): Device for export ('cpu', 'cuda', etc.).
                If None, defaults to 'cpu' for maximum compatibility.
            verbose (bool): Print export details (default: False).

        Returns:
            bool: True if export succeeded, False otherwise.

        Example:
            >>> trainer = SeqTrainer(hstu_model, ...)
            >>> trainer.fit(train_dl, val_dl)
            >>> trainer.export_onnx("hstu.onnx", vocab_size=10000)

            >>> # Export on specific device
            >>> trainer.export_onnx("hstu.onnx", vocab_size=10000, device="cpu")
        """
        import warnings

        # Use provided device or default to 'cpu'
        export_device = device if device is not None else 'cpu'

        # Handle DataParallel wrapped model
        model = self.model.module if hasattr(self.model, 'module') else self.model
        model.eval()
        model.to(export_device)

        # Get vocab_size from model if not provided
        if vocab_size is None:
            if hasattr(model, 'vocab_size'):
                vocab_size = model.vocab_size
            elif hasattr(model, 'item_num'):
                vocab_size = model.item_num
            else:
                raise ValueError("vocab_size must be provided or model must have 'vocab_size' or 'item_num' attribute")

        # Generate dummy inputs on the export device
        dummy_seq_tokens = torch.randint(0, vocab_size, (batch_size, seq_length), device=export_device)
        dummy_seq_time_diffs = torch.zeros(batch_size, seq_length, dtype=torch.float32, device=export_device)

        # Configure dynamic axes
        dynamic_axes = None
        if dynamic_batch:
            dynamic_axes = {"seq_tokens": {0: "batch_size", 1: "seq_length"}, "seq_time_diffs": {0: "batch_size", 1: "seq_length"}, "output": {0: "batch_size", 1: "seq_length"}}

        # Ensure output directory exists
        import os
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        try:
            with torch.no_grad():
                torch.onnx.export(
                    model,
                    (dummy_seq_tokens,
                     dummy_seq_time_diffs),
                    output_path,
                    input_names=["seq_tokens",
                                 "seq_time_diffs"],
                    output_names=["output"],
                    dynamic_axes=dynamic_axes,
                    opset_version=opset_version,
                    do_constant_folding=True,
                    verbose=verbose,
                    dynamo=False  # Use legacy exporter for dynamic_axes support
                )

            if verbose:
                print(f"Successfully exported ONNX model to: {output_path}")
                print("  Input names: ['seq_tokens', 'seq_time_diffs']")
                print(f"  Vocab size: {vocab_size}")
                print(f"  Opset version: {opset_version}")
                print(f"  Dynamic batch: {dynamic_batch}")

            return True

        except Exception as e:
            warnings.warn(f"ONNX export failed: {str(e)}")
            raise RuntimeError(f"Failed to export ONNX model: {str(e)}") from e
