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

    def __init__(
        self,
        model,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=None,
        scheduler_fn=None,
        scheduler_params=None,
        n_epoch=10,
        earlystop_patience=10,
        device='cpu',
        gpus=None,
        model_path='./',
        loss_type='cross_entropy',
        loss_params=None,
        model_logger=None
    ):
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

        self.loss_type = loss_type
        self.n_epoch = n_epoch
        self.early_stopper = EarlyStopper(patience=earlystop_patience)
        self.model_path = model_path
        self.model_logger = model_logger

    def fit(self, train_dataloader, val_dataloader=None):
        """训练模型.

        Args:
            train_dataloader (DataLoader): 训练数据加载器
            val_dataloader (DataLoader): 验证数据加载器

        Returns:
            dict: 训练历史
        """
        history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}

        for logger in self._iter_loggers():
            logger.log_hyperparams({'n_epoch': self.n_epoch, 'learning_rate': self.optimizer.param_groups[0]['lr'], 'loss_type': self.loss_type})

        for epoch_i in range(self.n_epoch):
            print('epoch:', epoch_i)
            # 训练阶段
            train_loss = self.train_one_epoch(train_dataloader)
            history['train_loss'].append(train_loss)

            # Collect metrics
            logs = {'train/loss': train_loss, 'learning_rate': self.optimizer.param_groups[0]['lr']}

            if self.scheduler is not None:
                if epoch_i % self.scheduler.step_size == 0:
                    print("Current lr : {}".format(self.optimizer.state_dict()['param_groups'][0]['lr']))
                self.scheduler.step()  # update lr in epoch level by scheduler

            # 验证阶段
            if val_dataloader:
                val_loss, val_accuracy = self.evaluate(val_dataloader)
                history['val_loss'].append(val_loss)
                history['val_accuracy'].append(val_accuracy)

                logs['val/loss'] = val_loss
                logs['val/accuracy'] = val_accuracy
                logs['auc'] = val_accuracy  # For compatibility with EarlyStopper

                print(f"epoch: {epoch_i}, validation: loss: {val_loss:.4f}, accuracy: {val_accuracy:.4f}")

                # 早停
                if self.early_stopper.stop_training(val_accuracy, self.model.state_dict()):
                    print(f'validation: best accuracy: {self.early_stopper.best_auc}')
                    self.model.load_state_dict(self.early_stopper.best_weights)
                    break

            for logger in self._iter_loggers():
                logger.log_metrics(logs, step=epoch_i)

        torch.save(self.model.state_dict(), os.path.join(self.model_path, "model.pth"))  # save best model

        for logger in self._iter_loggers():
            logger.finish()

        return history

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
        epoch_loss = 0
        batch_count = 0
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
            epoch_loss += loss.item()
            batch_count += 1
            if (i + 1) % log_interval == 0:
                tk0.set_postfix(loss=total_loss / log_interval)
                total_loss = 0

        # Return average epoch loss
        return epoch_loss / batch_count if batch_count > 0 else 0

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

    def export_onnx(self, output_path, batch_size=2, seq_length=50, vocab_size=None, opset_version=14, dynamic_batch=True, device=None, verbose=False, onnx_export_kwargs=None):
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
            onnx_export_kwargs (dict, optional): Extra kwargs forwarded to ``torch.onnx.export``.

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
                import inspect

                export_kwargs = {
                    "f": output_path,
                    "input_names": ["seq_tokens",
                                    "seq_time_diffs"],
                    "output_names": ["output"],
                    "dynamic_axes": dynamic_axes,
                    "opset_version": opset_version,
                    "do_constant_folding": True,
                    "verbose": verbose,
                }

                if onnx_export_kwargs:
                    overlap = set(export_kwargs.keys()) & set(onnx_export_kwargs.keys())
                    overlap.discard("dynamo")
                    if overlap:
                        raise ValueError("onnx_export_kwargs contains keys that overlap with explicit args: "
                                         f"{sorted(overlap)}. Please set them via export_onnx() parameters instead.")
                    export_kwargs.update(onnx_export_kwargs)

                # Auto-pick exporter:
                # - dynamic_axes present => prefer legacy exporter (dynamo=False) for dynamic batch/seq
                # - otherwise prefer dynamo exporter (dynamo=True) on newer torch
                sig = inspect.signature(torch.onnx.export)
                if "dynamo" in sig.parameters:
                    if "dynamo" not in export_kwargs:
                        export_kwargs["dynamo"] = False if dynamic_axes is not None else True
                else:
                    export_kwargs.pop("dynamo", None)

                torch.onnx.export(model, (dummy_seq_tokens, dummy_seq_time_diffs), **export_kwargs)

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

    def visualization(self, seq_length=50, vocab_size=None, batch_size=2, depth=3, show_shapes=True, expand_nested=True, save_path=None, graph_name="model", device=None, dpi=300, **kwargs):
        """Visualize the model's computation graph.

        This method generates a visual representation of the sequence model
        architecture, showing layer connections, tensor shapes, and nested
        module structures.

        Parameters
        ----------
        seq_length : int, default=50
            Sequence length for dummy input.
        vocab_size : int, optional
            Vocabulary size for generating dummy tokens.
            If None, will try to get from model.vocab_size or model.item_num.
        batch_size : int, default=2
            Batch size for dummy input.
        depth : int, default=3
            Visualization depth, higher values show more detail.
            Set to -1 to show all layers.
        show_shapes : bool, default=True
            Whether to display tensor shapes.
        expand_nested : bool, default=True
            Whether to expand nested modules.
        save_path : str, optional
            Path to save the graph image (.pdf, .svg, .png).
            If None, displays in Jupyter or opens system viewer.
        graph_name : str, default="model"
            Name for the graph.
        device : str, optional
            Device for model execution. If None, defaults to 'cpu'.
        dpi : int, default=300
            Resolution in dots per inch for output image.
            Higher values produce sharper images suitable for papers.
        **kwargs : dict
            Additional arguments passed to torchview.draw_graph().

        Returns
        -------
        ComputationGraph
            A torchview ComputationGraph object.

        Raises
        ------
        ImportError
            If torchview or graphviz is not installed.
        ValueError
            If vocab_size is not provided and cannot be inferred from model.

        Notes
        -----
        Default Display Behavior:
            When `save_path` is None (default):
            - In Jupyter/IPython: automatically displays the graph inline
            - In Python script: opens the graph with system default viewer

        Examples
        --------
        >>> trainer = SeqTrainer(hstu_model, ...)
        >>> trainer.fit(train_dl, val_dl)
        >>>
        >>> # Auto-display in Jupyter (no save_path needed)
        >>> trainer.visualization(depth=4, vocab_size=10000)
        >>>
        >>> # Save to high-DPI PNG for papers
        >>> trainer.visualization(save_path="model.png", dpi=300)
        """
        try:
            from torchview import draw_graph
            TORCHVIEW_AVAILABLE = True
        except ImportError:
            TORCHVIEW_AVAILABLE = False

        if not TORCHVIEW_AVAILABLE:
            raise ImportError(
                "Visualization requires torchview. "
                "Install with: pip install torch-rechub[visualization]\n"
                "Also ensure graphviz is installed on your system:\n"
                "  - Ubuntu/Debian: sudo apt-get install graphviz\n"
                "  - macOS: brew install graphviz\n"
                "  - Windows: choco install graphviz"
            )

        from ..utils.visualization import _is_jupyter_environment, display_graph

        # Handle DataParallel wrapped model
        model = self.model.module if hasattr(self.model, 'module') else self.model

        # Use provided device or default to 'cpu'
        viz_device = device if device is not None else 'cpu'

        # Get vocab_size from model if not provided
        if vocab_size is None:
            if hasattr(model, 'vocab_size'):
                vocab_size = model.vocab_size
            elif hasattr(model, 'item_num'):
                vocab_size = model.item_num
            else:
                raise ValueError("vocab_size must be provided or model must have "
                                 "'vocab_size' or 'item_num' attribute")

        # Generate dummy inputs for sequence model
        dummy_seq_tokens = torch.randint(0, vocab_size, (batch_size, seq_length), device=viz_device)
        dummy_seq_time_diffs = torch.zeros(batch_size, seq_length, dtype=torch.float32, device=viz_device)

        # Move model to device
        model = model.to(viz_device)
        model.eval()

        # Call torchview.draw_graph
        graph = draw_graph(model, input_data=(dummy_seq_tokens, dummy_seq_time_diffs), graph_name=graph_name, depth=depth, device=viz_device, expand_nested=expand_nested, show_shapes=show_shapes, save_graph=False, **kwargs)

        # Set DPI for high-quality output
        graph.visual_graph.graph_attr['dpi'] = str(dpi)

        # Handle save_path: manually save with DPI applied
        if save_path:
            import os
            directory = os.path.dirname(save_path) or "."
            filename = os.path.splitext(os.path.basename(save_path))[0]
            ext = os.path.splitext(save_path)[1].lstrip('.')
            output_format = ext if ext else 'pdf'
            if directory != "." and not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
            graph.visual_graph.render(filename=filename, directory=directory, format=output_format, cleanup=True)

        # Handle default display behavior when save_path is None
        if save_path is None:
            if _is_jupyter_environment():
                display_graph(graph)
            else:
                graph.visual_graph.view(cleanup=True)

        return graph
