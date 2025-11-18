"""Utility classes and functions for the HSTU model."""

import numpy as np
import torch
import torch.nn as nn


class RelPosBias(nn.Module):
    """Relative position bias module.

    This module is used in HSTU self-attention layers to provide a learnable
    bias that depends on the relative distance between sequence positions. It
    can be combined with time-based bucketing when needed.

    Args:
        n_heads (int): Number of attention heads.
        max_seq_len (int): Maximum supported sequence length.
        num_buckets (int): Number of relative position buckets. Default: 32.

    Shape:
        - Output: ``(1, n_heads, seq_len, seq_len)``

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
        self.rel_pos_bias_table = nn.Parameter(torch.randn(num_buckets, n_heads))

    def _relative_position_bucket(self, relative_position):
        """Map relative positions to bucket indices.

        Args:
            relative_position (Tensor): Relative position tensor ``(L, L)``.

        Returns:
            Tensor: Integer bucket indices with the same ``(L, L)`` shape.
        """
        num_buckets = self.num_buckets
        max_distance = self.max_seq_len

        # Use absolute distance and linearly map it to bucket indices
        relative_position = torch.abs(relative_position)

        bucket = torch.clamp(
            relative_position * (num_buckets - 1) // max_distance,
            0,
            num_buckets - 1,
        )

        return bucket.long()

    def forward(self, seq_len):
        """Compute relative position bias for a given sequence length.

        Args:
            seq_len (int): Sequence length ``L``.

        Returns:
            Tensor: Relative position bias of shape ``(1, n_heads, L, L)``.
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
    """Vocabulary mask used to constrain generation during inference.

    At inference time this module can be used to mask out invalid item IDs
    so that the model never generates them.

    Args:
        vocab_size (int): Vocabulary size.
        invalid_items (list, optional): List of invalid item IDs to be masked.

    Methods:
        apply_mask: Apply the mask to logits.

    Example:
        >>> mask = VocabMask(vocab_size=1000, invalid_items=[0, 1, 2])
        >>> logits = torch.randn(32, 1000)
        >>> masked_logits = mask.apply_mask(logits)
    """

    def __init__(self, vocab_size, invalid_items=None):
        super().__init__()
        self.vocab_size = vocab_size

        # Create a boolean mask over the vocabulary
        self.register_buffer(
            'mask',
            torch.ones(vocab_size,
                       dtype=torch.bool),
        )

        # Mark invalid items
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
    """Simple mapper between ``item_id`` and ``token_id``.

    In sequence generation tasks we often treat item IDs as tokens. This
    helper keeps a trivial identity mapping but makes the intent explicit and
    allows future extensions (e.g., reserved IDs, remapping, etc.).

    Args:
        vocab_size (int): Size of the vocabulary.
        pad_id (int): ID used for the PAD token. Default: 0.
        unk_id (int): ID used for unknown tokens. Default: 1.

    Methods:
        encode: Map ``item_id`` to ``token_id``.
        decode: Map ``token_id`` back to ``item_id``.

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
        token_ids = np.where((item_ids >= 0) & (item_ids < self.vocab_size), item_ids, self.unk_id)
        return token_ids

    def decode(self, token_ids):
        """将token_id转换为item_id.
        
        Args:
            token_ids (np.ndarray): token ID数组
            
        Returns:
            np.ndarray: item ID数组
        """
        # 处理超出范围的token_id
        item_ids = np.where((token_ids >= 0) & (token_ids < self.vocab_size), token_ids, self.unk_id)
        return item_ids
