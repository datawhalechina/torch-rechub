"""HSTU: Hierarchical Sequential Transduction Units Model."""

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
        
    Shape:
        - Input: `(batch_size, seq_len)`
        - Output: `(batch_size, seq_len, vocab_size)`
        
    Example:
        >>> model = HSTUModel(vocab_size=100000, d_model=512)
        >>> x = torch.randint(0, 100000, (32, 256))
        >>> logits = model(x)
        >>> logits.shape
        torch.Size([32, 256, 100000])
    """
    
    def __init__(self, vocab_size, d_model=512, n_heads=8, n_layers=4,
                 dqk=64, dv=64, max_seq_len=256, dropout=0.1, 
                 use_rel_pos_bias=True):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        
        # Token Embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        
        # Position Embedding
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        
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
    
    def forward(self, x):
        """前向传播.
        
        Args:
            x (Tensor): 输入token序列，shape: (batch_size, seq_len)
            
        Returns:
            Tensor: 输出logits，shape: (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = x.shape
        
        # Token Embedding
        token_emb = self.token_embedding(x)  # (B, L, D)
        
        # Position Embedding
        positions = torch.arange(seq_len, dtype=torch.long, device=x.device)
        pos_emb = self.position_embedding(positions)  # (L, D)
        
        # 合并token和position embedding
        embeddings = token_emb + pos_emb.unsqueeze(0)  # (B, L, D)
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

