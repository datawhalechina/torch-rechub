"""Sequence Generation Model Trainer."""

import os

import torch
import torch.nn as nn
import tqdm

from ..basic.callback import EarlyStopper


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
        model_path='./'
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
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=0)

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
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': []
        }

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
        """训练一个epoch.

        Args:
            data_loader (DataLoader): 训练数据加载器
            log_interval (int): 日志打印间隔

        Returns:
            float: 平均损失
        """
        self.model.train()
        total_loss = 0
        tk0 = tqdm.tqdm(data_loader, desc="train", smoothing=0, mininterval=1.0)
        for i, (seq_tokens, seq_positions, seq_time_diffs, targets) in enumerate(tk0):
            # 移动到设备
            seq_tokens = seq_tokens.to(self.device)
            seq_positions = seq_positions.to(self.device)
            seq_time_diffs = seq_time_diffs.to(self.device)
            targets = targets.to(self.device).squeeze(-1)

            # 前向传播
            logits = self.model(seq_tokens, seq_time_diffs)  # (B, L, V)

            # 计算损失
            # 对于next-item prediction任务，只使用最后一个位置的预测
            # logits[:, -1, :] 表示取每个序列的最后一个位置的预测
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
        """评估模型.

        Args:
            data_loader (DataLoader): 验证数据加载器

        Returns:
            tuple: (平均损失, 准确率)
        """
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for seq_tokens, seq_positions, seq_time_diffs, targets in tqdm.tqdm(data_loader, desc="evaluating", smoothing=0, mininterval=1.0):
                # 移动到设备
                seq_tokens = seq_tokens.to(self.device)
                seq_positions = seq_positions.to(self.device)
                seq_time_diffs = seq_time_diffs.to(self.device)
                targets = targets.to(self.device).squeeze(-1)

                # 前向传播
                logits = self.model(seq_tokens, seq_time_diffs)  # (B, L, V)

                # 计算损失
                # 对于next-item prediction任务，只使用最后一个位置的预测
                last_logits = logits[:, -1, :]  # (B, V)

                loss = self.loss_fn(last_logits, targets)
                total_loss += loss.item()

                # 计算准确率
                predictions = torch.argmax(last_logits, dim=-1)  # (B,)
                correct = (predictions == targets).sum().item()
                total_correct += correct
                total_samples += targets.numel()

        avg_loss = total_loss / len(data_loader)
        accuracy = total_correct / total_samples

        return avg_loss, accuracy

