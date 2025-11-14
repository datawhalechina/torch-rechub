"""HSTU模型的工具函数和类."""

import torch
import torch.nn as nn
import numpy as np


class RelPosBias(nn.Module):
    """相对位置偏置模块.
    
    用于HSTU层中的自注意力计算，支持时间bucketing。
    相对位置偏置可以帮助模型更好地捕捉序列中的相对位置信息。
    
    Args:
        n_heads (int): 多头注意力的头数
        max_seq_len (int): 最大序列长度
        num_buckets (int): 时间bucket的数量，默认32
        
    Shape:
        - Output: `(1, n_heads, seq_len, seq_len)`
        
    Example:
        >>> rel_pos_bias = RelPosBias(n_heads=8, max_seq_len=256)
        >>> bias = rel_pos_bias(256)
        >>> bias.shape
        torch.Size([1, 8, 256, 256])
    """
    
    def __init__(self, n_heads, max_seq_len, num_buckets=32):
        super().__init__()
        self.n_heads = n_heads
        self.max_seq_len = max_seq_len
        self.num_buckets = num_buckets
        
        # 相对位置偏置表: (num_buckets, n_heads)
        self.rel_pos_bias_table = nn.Parameter(
            torch.randn(num_buckets, n_heads)
        )
        
    def _relative_position_bucket(self, relative_position):
        """将相对位置映射到bucket.
        
        Args:
            relative_position (Tensor): 相对位置张量
            
        Returns:
            Tensor: bucket索引
        """
        num_buckets = self.num_buckets
        max_distance = self.max_seq_len
        
        # 处理负数位置
        relative_position = torch.abs(relative_position)
        
        # 线性映射到bucket
        bucket = torch.clamp(
            relative_position * (num_buckets - 1) // max_distance,
            0,
            num_buckets - 1
        )
        
        return bucket.long()
    
    def forward(self, seq_len):
        """计算相对位置偏置.
        
        Args:
            seq_len (int): 序列长度
            
        Returns:
            Tensor: 相对位置偏置，shape: (1, n_heads, seq_len, seq_len)
        """
        # 创建位置索引
        positions = torch.arange(seq_len, dtype=torch.long, device=self.rel_pos_bias_table.device)
        
        # 计算相对位置: (seq_len, seq_len)
        relative_positions = positions.unsqueeze(0) - positions.unsqueeze(1)
        
        # 映射到bucket
        buckets = self._relative_position_bucket(relative_positions)
        
        # 查表获取偏置: (seq_len, seq_len, n_heads)
        bias = self.rel_pos_bias_table[buckets]
        
        # 转置为 (1, n_heads, seq_len, seq_len)
        bias = bias.permute(2, 0, 1).unsqueeze(0)
        
        return bias


class VocabMask(nn.Module):
    """词表掩码模块，用于推理时约束生成.
    
    在推理时，可以通过掩码来排除某些无效的item，
    确保生成的item都是有效的。
    
    Args:
        vocab_size (int): 词表大小
        invalid_items (list, optional): 无效item的列表
        
    Methods:
        apply_mask: 应用掩码到logits
        
    Example:
        >>> mask = VocabMask(vocab_size=1000, invalid_items=[0, 1, 2])
        >>> logits = torch.randn(32, 1000)
        >>> masked_logits = mask.apply_mask(logits)
    """
    
    def __init__(self, vocab_size, invalid_items=None):
        super().__init__()
        self.vocab_size = vocab_size
        
        # 创建掩码
        self.register_buffer(
            'mask',
            torch.ones(vocab_size, dtype=torch.bool)
        )
        
        # 标记无效item
        if invalid_items is not None:
            for item_id in invalid_items:
                if 0 <= item_id < vocab_size:
                    self.mask[item_id] = False
    
    def apply_mask(self, logits):
        """应用掩码到logits.
        
        Args:
            logits (Tensor): 模型输出logits，shape: (..., vocab_size)
            
        Returns:
            Tensor: 掩码后的logits
        """
        # 将无效item的logits设置为极小值
        masked_logits = logits.clone()
        masked_logits[..., ~self.mask] = -1e9
        
        return masked_logits


class VocabMapper(object):
    """词表映射器，处理item_id和token_id的转换.
    
    在序列生成任务中，需要将item_id映射到token_id，
    以及将token_id映射回item_id。
    
    Args:
        vocab_size (int): 词表大小
        pad_id (int): PAD token的ID，默认0
        unk_id (int): UNK token的ID，默认1
        
    Methods:
        encode: item_id -> token_id
        decode: token_id -> item_id
        
    Example:
        >>> mapper = VocabMapper(vocab_size=1000)
        >>> item_ids = np.array([10, 20, 30])
        >>> token_ids = mapper.encode(item_ids)
        >>> decoded_ids = mapper.decode(token_ids)
    """
    
    def __init__(self, vocab_size, pad_id=0, unk_id=1):
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_id = pad_id
        self.unk_id = unk_id
        
        # 创建映射表（简单的恒等映射）
        self.item2token = np.arange(vocab_size)
        self.token2item = np.arange(vocab_size)
    
    def encode(self, item_ids):
        """将item_id转换为token_id.
        
        Args:
            item_ids (np.ndarray): item ID数组
            
        Returns:
            np.ndarray: token ID数组
        """
        # 处理超出范围的item_id
        token_ids = np.where(
            (item_ids >= 0) & (item_ids < self.vocab_size),
            item_ids,
            self.unk_id
        )
        return token_ids
    
    def decode(self, token_ids):
        """将token_id转换为item_id.
        
        Args:
            token_ids (np.ndarray): token ID数组
            
        Returns:
            np.ndarray: item ID数组
        """
        # 处理超出范围的token_id
        item_ids = np.where(
            (token_ids >= 0) & (token_ids < self.vocab_size),
            token_ids,
            self.unk_id
        )
        return item_ids

