"""Sequence Generation Model Trainer."""

import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np


class SeqTrainer(object):
    """序列生成模型训练器.
    
    用于训练HSTU等序列生成模型。
    支持CrossEntropyLoss损失函数和生成式评估指标。
    
    Args:
        model (nn.Module): 要训练的模型
        optimizer (Optimizer): 优化器
        device (str): 设备，'cpu'或'cuda'，默认'cpu'
        
    Methods:
        fit: 训练模型
        evaluate: 评估模型
        predict: 生成预测
        
    Example:
        >>> trainer = SeqTrainer(
        ...     model=model,
        ...     optimizer=torch.optim.Adam(model.parameters()),
        ...     device='cuda'
        ... )
        >>> trainer.fit(
        ...     train_loader=train_loader,
        ...     val_loader=val_loader,
        ...     epochs=20
        ... )
    """
    
    def __init__(self, model, optimizer, device='cpu'):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        
        # 损失函数
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=0)
        
        # 移动模型到设备
        self.model.to(device)
    
    def fit(self, train_loader, val_loader=None, epochs=20, 
            early_stopping_patience=3, save_path=None):
        """训练模型.
        
        Args:
            train_loader (DataLoader): 训练数据加载器
            val_loader (DataLoader): 验证数据加载器
            epochs (int): 训练轮数，默认20
            early_stopping_patience (int): 早停耐心值，默认3
            save_path (str): 模型保存路径
            
        Returns:
            dict: 训练历史
        """
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': []
        }
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # 训练阶段
            train_loss = self._train_epoch(train_loader)
            history['train_loss'].append(train_loss)
            
            # 验证阶段
            if val_loader is not None:
                val_loss, val_accuracy = self.evaluate(val_loader)
                history['val_loss'].append(val_loss)
                history['val_accuracy'].append(val_accuracy)
                
                print(f"Epoch {epoch+1}/{epochs} - "
                      f"train_loss: {train_loss:.4f}, "
                      f"val_loss: {val_loss:.4f}, "
                      f"val_accuracy: {val_accuracy:.4f}")
                
                # 早停
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    
                    # 保存最佳模型
                    if save_path is not None:
                        torch.save(self.model.state_dict(), save_path)
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        break
            else:
                print(f"Epoch {epoch+1}/{epochs} - train_loss: {train_loss:.4f}")
        
        return history
    
    def _train_epoch(self, train_loader):
        """训练一个epoch.
        
        Args:
            train_loader (DataLoader): 训练数据加载器
            
        Returns:
            float: 平均损失
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in tqdm(train_loader, desc="Training"):
            seq_tokens, seq_positions, targets = batch
            
            # 移动到设备
            seq_tokens = seq_tokens.to(self.device)
            seq_positions = seq_positions.to(self.device)
            targets = targets.to(self.device).squeeze(-1)
            
            # 前向传播
            logits = self.model(seq_tokens)  # (B, L, V)
            
            # 计算损失
            # 需要将logits和targets reshape为 (B*L, V) 和 (B*L,)
            batch_size, seq_len, vocab_size = logits.shape
            logits_flat = logits.reshape(batch_size * seq_len, vocab_size)
            targets_flat = targets.reshape(batch_size * seq_len)
            
            loss = self.loss_fn(logits_flat, targets_flat)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def evaluate(self, val_loader):
        """评估模型.
        
        Args:
            val_loader (DataLoader): 验证数据加载器
            
        Returns:
            tuple: (平均损失, 准确率)
        """
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluating"):
                seq_tokens, seq_positions, targets = batch
                
                # 移动到设备
                seq_tokens = seq_tokens.to(self.device)
                seq_positions = seq_positions.to(self.device)
                targets = targets.to(self.device).squeeze(-1)
                
                # 前向传播
                logits = self.model(seq_tokens)  # (B, L, V)
                
                # 计算损失
                batch_size, seq_len, vocab_size = logits.shape
                logits_flat = logits.reshape(batch_size * seq_len, vocab_size)
                targets_flat = targets.reshape(batch_size * seq_len)
                
                loss = self.loss_fn(logits_flat, targets_flat)
                total_loss += loss.item()
                
                # 计算准确率
                predictions = torch.argmax(logits_flat, dim=-1)
                correct = (predictions == targets_flat).sum().item()
                total_correct += correct
                total_samples += targets_flat.numel()
        
        avg_loss = total_loss / len(val_loader)
        accuracy = total_correct / total_samples
        
        return avg_loss, accuracy

