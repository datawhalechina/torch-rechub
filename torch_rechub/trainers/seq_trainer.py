"""Sequence Generation Model Trainer."""

import os
from numbers import Integral

import torch
import torch.nn as nn
import torch.nn.functional as F
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
        loss_type (str): cross_entropy (default), legacy nce (full vocabulary),
            or sampled_softmax (positive plus sampled catalog negatives).
        loss_params (dict): For sampled_softmax: num_negatives=128 and optional
            negative_similarity_threshold for normalized HLLM item vectors.
        accumulation_steps (int): Microbatches per update, default 1.
        amp (bool): CUDA float16 autocast/scaling, disabled by default.
        max_grad_norm (float, optional): Clip unscaled gradients.
        eval_chunk_size (int): Candidate scoring chunk size, default 4096.

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
        model_logger=None,
        accumulation_steps=1,
        amp=False,
        max_grad_norm=None,
        eval_chunk_size=4096,
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
        if not isinstance(accumulation_steps, Integral) or accumulation_steps < 1:
            raise ValueError("accumulation_steps must be a positive integer")
        if not isinstance(eval_chunk_size, Integral) or eval_chunk_size < 1:
            raise ValueError("eval_chunk_size must be a positive integer")
        if max_grad_norm is not None and max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive")
        self.accumulation_steps = accumulation_steps
        self.amp = bool(amp and self.device.type == 'cuda')
        # Small-temperature cosine scores amplify gradients; a modest initial
        # scale avoids skipping all updates in short tutorial runs.
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.amp, init_scale=1024) if hasattr(torch, 'amp') and hasattr(torch.amp, 'GradScaler') else torch.cuda.amp.GradScaler(enabled=self.amp, init_scale=1024)
        self.max_grad_norm = max_grad_norm
        self.eval_chunk_size = eval_chunk_size
        if optimizer_params is None:
            optimizer_params = {"lr": 1e-3, "weight_decay": 1e-5}
        self.optimizer = optimizer_fn(self.model.parameters(), **optimizer_params)  # default optimizer
        self.scheduler = None
        if scheduler_fn is not None:
            self.scheduler = scheduler_fn(self.optimizer, **(scheduler_params or {}))

        # 损失函数
        if loss_type == 'sampled_softmax':
            params = dict(loss_params or {})
            self.num_negatives = params.pop('num_negatives', 128)
            self.negative_similarity_threshold = params.pop('negative_similarity_threshold', None)
            if self.negative_similarity_threshold is not None:
                if not 0 <= self.negative_similarity_threshold <= 1 or not hasattr(self.model, 'get_item_embeddings'):
                    raise ValueError("negative_similarity_threshold requires normalized get_item_embeddings() and a value in [0, 1]")
            if params or not isinstance(self.num_negatives, Integral) or self.num_negatives < 1:
                raise ValueError("Unknown sampled_softmax parameters or invalid num_negatives")
            if len(gpus) > 1:
                raise ValueError("sampled_softmax currently supports one device; use the separate multi-GPU branch")
            if not all(hasattr(self.model, method) for method in ('encode', 'score_items', 'vocab_size')):
                raise ValueError("sampled_softmax requires encode(), score_items() and vocab_size")
            if self.model.vocab_size < 3:
                raise ValueError("sampled_softmax requires at least two real items plus PAD")
            self.loss_fn = nn.CrossEntropyLoss(ignore_index=0)
        elif loss_type == 'nce':
            if loss_params is None:
                loss_params = {"temperature": 0.1, "ignore_index": 0}
            self.loss_fn = NCELoss(**loss_params)
        elif loss_type == 'cross_entropy':
            if loss_params is None:
                loss_params = {"ignore_index": 0}
            self.loss_fn = nn.CrossEntropyLoss(**loss_params)
        else:
            raise ValueError("loss_type must be cross_entropy, nce or sampled_softmax")

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
        best_score, best_loss, best_weights = float('-inf'), float('inf'), None
        os.makedirs(self.model_path, exist_ok=True)
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau) and val_dataloader is None:
            raise ValueError("ReduceLROnPlateau requires validation data")

        for logger in self._iter_loggers():
            logger.log_hyperparams({'n_epoch': self.n_epoch, 'learning_rate': self.optimizer.param_groups[0]['lr'], 'loss_type': self.loss_type})

        for epoch_i in range(self.n_epoch):
            print('epoch:', epoch_i)
            # 训练阶段
            train_loss = self.train_one_epoch(train_dataloader)
            history['train_loss'].append(train_loss)

            # Collect metrics
            logs = {'train/loss': train_loss, 'learning_rate': self.optimizer.param_groups[0]['lr']}

            # 验证阶段
            if val_dataloader is not None:
                val_loss, val_accuracy = self.evaluate(val_dataloader)
                history['val_loss'].append(val_loss)
                history['val_accuracy'].append(val_accuracy)

                logs['val/loss'] = val_loss
                logs['val/accuracy'] = val_accuracy
                logs['auc'] = val_accuracy  # For compatibility with EarlyStopper

                print(f"epoch: {epoch_i}, validation: loss: {val_loss:.4f}, accuracy: {val_accuracy:.4f}")
                if val_accuracy > best_score or (val_accuracy == best_score and val_loss < best_loss):
                    best_score = val_accuracy
                    best_loss = val_loss
                    # Keep the best snapshot off GPU, especially for a frozen
                    # LLM item table. Loading copies it back to model devices.
                    best_weights = {name: value.detach().cpu().clone() for name, value in self.model.state_dict().items()}

                # 早停
                if self.early_stopper.stop_training(val_accuracy, best_weights):
                    print(f'validation: best accuracy: {self.early_stopper.best_auc}')
                    break

            for logger in self._iter_loggers():
                logger.log_metrics(logs, step=epoch_i)
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_accuracy if self.scheduler.mode == 'max' else val_loss)
            elif self.scheduler is not None:
                self.scheduler.step()

        if best_weights is not None:
            self.model.load_state_dict(best_weights)
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

    def _compute_next_token_loss(self, logits, seq_tokens, targets):
        """Cross-entropy over the whole sequence (autoregressive next-token).

        ``logits[:, i, :]`` predicts the following real token; the final
        non-PAD position predicts the held-out target. This works with left
        and right padding (not just the final physical array slot). PAD
        positions are ignored based on the *current* token. This matters for
        left-padded sequences: the last PAD slot may have the first real item as
        its shifted label, but the model should not learn a PAD -> item
        transition.
        """
        vocab_size = logits.size(-1)
        next_tokens = self._next_tokens(seq_tokens, targets)
        if vocab_size > 0:
            logits = logits.clone()
            logits[..., 0] = torch.finfo(logits.dtype).min
        return self.loss_fn(logits.reshape(-1, vocab_size), next_tokens.reshape(-1))

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
        self.optimizer.zero_grad()
        for i, (seq_tokens, _seq_positions, seq_time_diffs, targets) in enumerate(tk0):
            # Move tensors to the target device. ``seq_positions`` is unused by
            # ``HSTUModel`` (which derives positions internally); keep the
            # dataloader API stable but drop it here.
            seq_tokens = seq_tokens.to(self.device)
            seq_time_diffs = seq_time_diffs.to(self.device)
            targets = targets.to(self.device).reshape(-1)

            with torch.autocast(device_type=self.device.type, enabled=self.amp):
                if self.loss_type == 'sampled_softmax':
                    loss = self._sampled_loss(self.model.encode(seq_tokens, seq_time_diffs), seq_tokens, targets)
                else:
                    logits = self.model(seq_tokens, seq_time_diffs)
                    loss = self._compute_next_token_loss(logits, seq_tokens, targets)
            # Scale by the actual microbatch count, including the short tail.
            group_start = (i // self.accumulation_steps) * self.accumulation_steps
            group_size = min(self.accumulation_steps, len(data_loader) - group_start)
            self.scaler.scale(loss / group_size).backward()
            if (i + 1) % self.accumulation_steps == 0 or i + 1 == len(data_loader):
                self.scaler.unscale_(self.optimizer)
                if self.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            total_loss += loss.item()
            epoch_loss += loss.item()
            batch_count += 1
            if (i + 1) % log_interval == 0:
                tk0.set_postfix(loss=total_loss / log_interval)
                total_loss = 0

        return epoch_loss / batch_count if batch_count > 0 else 0

    def evaluate(self, data_loader):
        """Evaluate the model on a validation/test data loader.

        Loss matches training: full-vocabulary CE or sampled candidate CE.
        Sampled loss uses fixed validation draws and is not full-vocabulary CE.
        accuracy is top-1 hit on the held-out next item only (useful as a
        ranking-style proxy metric).

        Returns:
            tuple: ``(avg_loss, top1_accuracy)``.
        """
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        # Fixed validation candidates make loss comparisons reproducible and
        # do not consume the training sampler's random stream.
        generator = torch.Generator(device=self.device).manual_seed(0)

        with torch.no_grad():
            for seq_tokens, _seq_positions, seq_time_diffs, targets in tqdm.tqdm(data_loader, desc="evaluating", smoothing=0, mininterval=1.0):
                seq_tokens = seq_tokens.to(self.device)
                seq_time_diffs = seq_time_diffs.to(self.device)
                targets = targets.to(self.device).reshape(-1)

                if self.loss_type == 'sampled_softmax':
                    states = self.model.encode(seq_tokens, seq_time_diffs)
                    loss = self._sampled_loss(states, seq_tokens, targets, generator=generator)
                    last = self._last_positions(seq_tokens)
                    last_states = states[torch.arange(states.size(0), device=self.device), last]
                    predictions = self._top1(last_states)
                else:
                    logits = self.model(seq_tokens, seq_time_diffs)
                    loss = self._compute_next_token_loss(logits, seq_tokens, targets)
                    last_logits = logits[torch.arange(logits.size(0), device=self.device), self._last_positions(seq_tokens)].clone()
                    last_logits[:, 0] = torch.finfo(last_logits.dtype).min
                    predictions = torch.argmax(last_logits, dim=-1)
                total_loss += loss.item()
                total_correct += (predictions == targets).sum().item()
                total_samples += targets.numel()

        avg_loss = total_loss / max(1, len(data_loader))
        accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        return avg_loss, accuracy

    def _last_positions(self, seq_tokens):
        valid = seq_tokens.ne(0)
        if not valid.any(dim=1).all():
            raise ValueError("Every sequence must contain at least one non-PAD item")
        positions = torch.arange(seq_tokens.size(1), device=seq_tokens.device).expand_as(seq_tokens)
        last = positions.masked_fill(~valid, -1).max(dim=1).values
        first = positions.masked_fill(~valid, seq_tokens.size(1)).min(dim=1).values
        if not (last - first + 1).eq(valid.sum(dim=1)).all():
            raise ValueError("Use contiguous histories with left or right padding, not internal PAD gaps")
        return last

    def _next_tokens(self, seq_tokens, targets):
        if targets.numel() != seq_tokens.size(0) or targets.le(0).any():
            raise ValueError("Provide one non-PAD target per history")
        last = self._last_positions(seq_tokens)
        next_tokens = torch.cat([seq_tokens[:, 1:], torch.zeros_like(seq_tokens[:, :1])], dim=1)
        next_tokens.scatter_(1, last.unsqueeze(1), targets.reshape(-1, 1))
        return next_tokens.masked_fill(seq_tokens.eq(0), 0)

    def _sampled_loss(self, states, seq_tokens, targets, generator=None):
        labels = self._next_tokens(seq_tokens, targets)
        valid = labels.ne(0)
        positive_ids = labels[valid]
        states = states[valid]
        # Uniform catalog negatives shared across valid positions. Unlike the
        # reference's per-position draws, this trades diversity for efficiency.
        # Repeated positives are never treated as their own negatives.
        negatives = torch.randint(1, self.model.vocab_size, (self.num_negatives,), device=self.device, generator=generator)
        # Compute matched positive scores in small blocks, not an N x N matrix
        # of every valid position against every positive in the batch.
        positive_scores = torch.cat([self.model.score_items(states[start:start + 64], positive_ids[start:start + 64]).diagonal() for start in range(0, states.size(0), 64)]).unsqueeze(1)
        negative_scores = self.model.score_items(states, negatives)
        negative_scores = negative_scores.masked_fill(positive_ids[:, None].eq(negatives[None, :]), torch.finfo(negative_scores.dtype).min)
        if self.negative_similarity_threshold is not None:
            with torch.no_grad():
                similarity = self.model.get_item_embeddings(positive_ids) @ self.model.get_item_embeddings(negatives).t()
            negative_scores = negative_scores.masked_fill(similarity > self.negative_similarity_threshold, torch.finfo(negative_scores.dtype).min)
        # CE is computed in float32 for small-temperature/AMP stability.
        return F.cross_entropy(torch.cat([positive_scores, negative_scores], dim=1).float(), torch.zeros(states.size(0), dtype=torch.long, device=self.device))

    def _top1(self, states):
        best_scores = torch.full((states.size(0),), float('-inf'), device=self.device)
        best_ids = torch.ones(states.size(0), dtype=torch.long, device=self.device)
        for start in range(1, self.model.vocab_size, self.eval_chunk_size):
            item_ids = torch.arange(start, min(start + self.eval_chunk_size, self.model.vocab_size), device=self.device)
            scores, indices = self.model.score_items(states, item_ids).max(dim=-1)
            improved = scores > best_scores
            best_scores = torch.where(improved, scores, best_scores)
            best_ids = torch.where(improved, item_ids[indices], best_ids)
        return best_ids

    @torch.no_grad()
    def predict(self, data_loader, top_k=10, filter_seen=False):
        """Return exact catalog top-k IDs with bounded scoring memory.

        PAD is always excluded. History filtering is opt-in because a repeated
        interaction may be the true next item. This is not sampled evaluation.
        """
        model = self.model.module if hasattr(self.model, 'module') else self.model
        if not isinstance(top_k, Integral) or top_k < 1:
            raise ValueError("top_k must be a positive integer")
        top_k = min(top_k, model.vocab_size - 1)
        was_training = self.model.training
        self.model.eval()
        predictions = []
        try:
            for seq_tokens, _positions, time_diffs, _targets in data_loader:
                seq_tokens, time_diffs = seq_tokens.to(self.device), time_diffs.to(self.device)
                states = model.encode(seq_tokens, time_diffs)
                states = states[torch.arange(states.size(0), device=self.device), self._last_positions(seq_tokens)]
                best_scores = torch.empty(states.size(0), 0, device=self.device)
                best_ids = torch.empty(states.size(0), 0, device=self.device, dtype=torch.long)
                for start in range(1, model.vocab_size, self.eval_chunk_size):
                    ids = torch.arange(start, min(start + self.eval_chunk_size, model.vocab_size), device=self.device)
                    scores = model.score_items(states, ids)
                    if filter_seen:
                        seen = torch.zeros(states.size(0), ids.size(0), dtype=torch.bool, device=self.device)
                        # Mark history IDs only within the current catalog chunk.
                        rows, positions = torch.where((seq_tokens >= start) & (seq_tokens < start + ids.size(0)))
                        seen[rows, seq_tokens[rows, positions] - start] = True
                        scores = scores.masked_fill(seen, float('-inf'))
                    scores = torch.cat([best_scores, scores], dim=-1)
                    candidates = torch.cat([best_ids, ids.expand(states.size(0), -1)], dim=-1)
                    best_scores, positions = scores.topk(min(top_k, scores.size(1)), dim=-1)
                    best_ids = candidates.gather(1, positions)
                # Not enough eligible items: -1, never a seen item pretending
                # to be a legal recommendation.
                best_ids = best_ids.masked_fill(~torch.isfinite(best_scores), -1)
                predictions.append(best_ids.cpu())
        finally:
            self.model.train(was_training)
        return torch.cat(predictions) if predictions else torch.empty(0, top_k, dtype=torch.long)

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
        original_device = next(model.parameters()).device
        original_training = model.training

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
            model.to(export_device)
            model.eval()
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
        finally:
            # Export is an inspection step, not a permanent device/mode switch.
            model.to(original_device)
            model.train(original_training)

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

        original_device = next(model.parameters()).device
        original_training = model.training
        try:
            model.to(viz_device)
            model.eval()
            graph = draw_graph(model, input_data=(dummy_seq_tokens, dummy_seq_time_diffs), graph_name=graph_name, depth=depth, device=viz_device, expand_nested=expand_nested, show_shapes=show_shapes, save_graph=False, **kwargs)
        finally:
            # A CPU diagram must not move the next CUDA training batch to CPU.
            model.to(original_device)
            model.train(original_training)

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
