"""HSTU: Hierarchical Sequential Transduction Units Model."""

import math
import torch
import torch.nn as nn
from torch_rechub.basic.layers import HSTUBlock
from torch_rechub.utils.hstu_utils import RelPosBias


class HSTUModel(nn.Module):
    """HSTU: Hierarchical Sequential Transduction Units.

    生成式推荐模型，用于序列生成任务。
    该模型通过多层HSTU块来捕捉序列中的长期依赖关系，
    并生成下一个item的概率分布。

    Args:
        vocab_size (int): 词表大小
        d_model (int): 模型维度，默认512
        n_heads (int): 多头注意力的头数，默认8
        n_layers (int): HSTU层的数量，默认4
        dqk (int): Query/Key的维度，默认64
        dv (int): Value的维度，默认64
        max_seq_len (int): 最大序列长度，默认256
        dropout (float): Dropout概率，默认0.1
        use_rel_pos_bias (bool): 是否使用相对位置偏置，默认True
        use_time_embedding (bool): 是否使用时间嵌入，默认True
        num_time_buckets (int): 时间bucket数量，默认2048
        time_bucket_fn (str): 时间bucket化函数，'sqrt'或'log'，默认'sqrt'

    Shape:
        - Input: `(batch_size, seq_len)` 或 `(batch_size, seq_len), (batch_size, seq_len)`
        - Output: `(batch_size, seq_len, vocab_size)`

    Example:
        >>> model = HSTUModel(vocab_size=100000, d_model=512)
        >>> x = torch.randint(0, 100000, (32, 256))
        >>> time_diffs = torch.randint(0, 86400, (32, 256))  # 时间差（秒）
        >>> logits = model(x, time_diffs)
        >>> logits.shape
        torch.Size([32, 256, 100000])
    """

    def __init__(self, vocab_size, d_model=512, n_heads=8, n_layers=4,
                 dqk=64, dv=64, max_seq_len=256, dropout=0.1,
                 use_rel_pos_bias=True, use_time_embedding=True,
                 num_time_buckets=2048, time_bucket_fn='sqrt'):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        self.use_time_embedding = use_time_embedding
        self.num_time_buckets = num_time_buckets
        self.time_bucket_fn = time_bucket_fn

        # Alpha缩放因子（参考Meta官方实现）
        self.alpha = math.sqrt(d_model)

        # Token Embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)

        # Position Embedding
        self.position_embedding = nn.Embedding(max_seq_len, d_model)

        # Time Embedding（新增）
        if use_time_embedding:
            # 时间嵌入表：将时间差映射到嵌入向量
            # num_time_buckets + 1: 包括padding的bucket
            self.time_embedding = nn.Embedding(num_time_buckets + 1, d_model, padding_idx=0)

        # HSTU Block
        self.hstu_block = HSTUBlock(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dqk=dqk,
            dv=dv,
            dropout=dropout,
            use_rel_pos_bias=use_rel_pos_bias
        )

        # 相对位置偏置
        self.use_rel_pos_bias = use_rel_pos_bias
        if use_rel_pos_bias:
            self.rel_pos_bias = RelPosBias(n_heads, max_seq_len)

        # 输出投影层
        self.output_projection = nn.Linear(d_model, vocab_size)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化模型权重."""
        for name, param in self.named_parameters():
            if 'weight' in name and len(param.shape) > 1:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)

    def _time_diff_to_bucket(self, time_diffs):
        """将时间差转换为bucket索引.

        参考Meta官方实现，使用sqrt或log函数将连续的时间差映射到离散的bucket。

        Args:
            time_diffs (Tensor): 时间差（秒），shape: (batch_size, seq_len)

        Returns:
            Tensor: bucket索引，shape: (batch_size, seq_len)
        """
        # 时间单位转换：秒 → 分钟（参考Meta官方实现）
        time_bucket_increments = 60.0
        time_diffs = time_diffs.float() / time_bucket_increments

        # 确保时间差非负，使用1e-6避免log(0)
        time_diffs = torch.clamp(time_diffs, min=1e-6)

        if self.time_bucket_fn == 'sqrt':
            # 使用平方根函数：适合时间差分布较均匀的情况
            # sqrt(time_diff / 60) 将时间差（分钟）压缩到较小的范围
            buckets = torch.sqrt(time_diffs).long()
        elif self.time_bucket_fn == 'log':
            # 使用对数函数：适合时间差跨度很大的情况
            # log(time_diff / 60) 避免log(0)
            buckets = torch.log(time_diffs).long()
        else:
            raise ValueError(f"不支持的time_bucket_fn: {self.time_bucket_fn}")

        # 限制bucket范围：[0, num_time_buckets-1]
        buckets = torch.clamp(buckets, min=0, max=self.num_time_buckets - 1)

        return buckets

    def forward(self, x, time_diffs=None):
        """前向传播.

        Args:
            x (Tensor): 输入token序列，shape: (batch_size, seq_len)
            time_diffs (Tensor, optional): 时间差序列（秒），shape: (batch_size, seq_len)
                如果为None且use_time_embedding=True，则使用全0的时间差

        Returns:
            Tensor: 输出logits，shape: (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = x.shape

        # Token Embedding（应用Alpha缩放，参考Meta官方实现）
        token_emb = self.token_embedding(x) * self.alpha  # (B, L, D)

        # Position Embedding
        positions = torch.arange(seq_len, dtype=torch.long, device=x.device)
        pos_emb = self.position_embedding(positions)  # (L, D)

        # 合并token和position embedding
        embeddings = token_emb + pos_emb.unsqueeze(0)  # (B, L, D)

        # Time Embedding（新增）
        if self.use_time_embedding:
            if time_diffs is None:
                # 如果没有提供时间差，使用全0（向后兼容）
                time_diffs = torch.zeros(batch_size, seq_len, dtype=torch.long, device=x.device)

            # 将时间差转换为bucket索引
            time_buckets = self._time_diff_to_bucket(time_diffs)  # (B, L)

            # 获取时间嵌入
            time_emb = self.time_embedding(time_buckets)  # (B, L, D)

            # 合并时间嵌入：embeddings = token_emb + pos_emb + time_emb
            embeddings = embeddings + time_emb

        embeddings = self.dropout(embeddings)

        # 计算相对位置偏置
        rel_pos_bias = None
        if self.use_rel_pos_bias:
            rel_pos_bias = self.rel_pos_bias(seq_len)  # (1, H, L, L)

        # HSTU Block
        hstu_output = self.hstu_block(embeddings, rel_pos_bias=rel_pos_bias)  # (B, L, D)

        # 输出投影
        logits = self.output_projection(hstu_output)  # (B, L, V)
        
        return logits

