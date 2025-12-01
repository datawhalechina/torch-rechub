"""HLLM: Hierarchical Large Language Model for Recommendation."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_rechub.utils.hstu_utils import RelPosBias


class HLLMTransformerBlock(nn.Module):
    """Single HLLM Transformer block with self-attention and FFN.
    
    This block is similar to HSTULayer but designed for HLLM which uses
    pre-computed item embeddings as input instead of learnable token embeddings.
    
    Args:
        d_model (int): Hidden dimension.
        n_heads (int): Number of attention heads.
        dropout (float): Dropout rate.
    """

    def __init__(self, d_model=512, n_heads=8, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim**-0.5

        # Multi-head self-attention
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

        # Feed-forward network
        ffn_hidden = 4 * d_model
        self.ffn = nn.Sequential(nn.Linear(d_model, ffn_hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(ffn_hidden, d_model), nn.Dropout(dropout))

        # Layer normalization and dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, rel_pos_bias=None):
        """Forward pass.
        
        Args:
            x (Tensor): Input of shape (B, L, D).
            rel_pos_bias (Tensor, optional): Relative position bias.
            
        Returns:
            Tensor: Output of shape (B, L, D).
        """
        batch_size, seq_len, _ = x.shape

        # Self-attention with residual
        residual = x
        x = self.norm1(x)

        Q = self.W_Q(x)  # (B, L, D)
        K = self.W_K(x)
        V = self.W_V(x)

        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        # Causal mask
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool))
        scores = scores.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        # Add relative position bias if provided
        if rel_pos_bias is not None:
            scores = scores + rel_pos_bias

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.d_model)
        attn_output = self.W_O(attn_output)
        attn_output = self.dropout(attn_output)

        x = residual + attn_output

        # FFN with residual
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + x

        return x


class HLLMModel(nn.Module):
    """HLLM: Hierarchical Large Language Model for Recommendation.

    This is a lightweight implementation of HLLM that uses pre-computed item
    embeddings as input. The original ByteDance HLLM uses end-to-end training
    with both Item LLM and User LLM, but this implementation focuses on the
    User LLM component for resource efficiency.

    Architecture:
        - Item Embeddings: Pre-computed using LLM (offline, frozen)
          Format: "{item_prompt}title: {title}description: {description}"
          where item_prompt = "Compress the following sentence into embedding: "
        - User LLM: Transformer blocks that model user sequences (trainable)
        - Scoring Head: Dot product between user representation and item embeddings

    Reference:
        ByteDance HLLM: https://github.com/bytedance/HLLM

    Args:
        item_embeddings (Tensor or str): Pre-computed item embeddings of shape
            (vocab_size, d_model), or path to a .pt file containing embeddings.
            Generated using the last token's hidden state from an LLM.
        vocab_size (int): Vocabulary size (number of items).
        d_model (int): Hidden dimension. Should match item embedding dimension.
            Default: 512. TinyLlama uses 2048, Baichuan2 uses 4096.
        n_heads (int): Number of attention heads. Default: 8.
        n_layers (int): Number of transformer blocks. Default: 4.
        max_seq_len (int): Maximum sequence length. Default: 256.
            Official uses MAX_ITEM_LIST_LENGTH=50.
        dropout (float): Dropout rate. Default: 0.1.
        use_rel_pos_bias (bool): Whether to use relative position bias. Default: True.
        use_time_embedding (bool): Whether to use time embeddings. Default: True.
        num_time_buckets (int): Number of time buckets. Default: 2048.
        time_bucket_fn (str): Time bucketization function ('sqrt' or 'log'). Default: 'sqrt'.
        temperature (float): Temperature for NCE scoring. Default: 1.0.
            Official uses logit_scale = log(1/0.07) ≈ 2.66.
    """

    def __init__(self, item_embeddings, vocab_size, d_model=512, n_heads=8, n_layers=4, max_seq_len=256, dropout=0.1, use_rel_pos_bias=True, use_time_embedding=True, num_time_buckets=2048, time_bucket_fn='sqrt', temperature=1.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        self.use_time_embedding = use_time_embedding
        self.num_time_buckets = num_time_buckets
        self.time_bucket_fn = time_bucket_fn
        self.temperature = temperature

        # Load item embeddings
        if isinstance(item_embeddings, str):
            item_embeddings = torch.load(item_embeddings)

        # Register as buffer (not trainable)
        self.register_buffer('item_embeddings', item_embeddings.float())

        # Positional embedding
        self.position_embedding = nn.Embedding(max_seq_len, d_model)

        # Time embedding
        if use_time_embedding:
            self.time_embedding = nn.Embedding(num_time_buckets + 1, d_model, padding_idx=0)

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([HLLMTransformerBlock(d_model=d_model, n_heads=n_heads, dropout=dropout) for _ in range(n_layers)])

        # Relative position bias
        self.use_rel_pos_bias = use_rel_pos_bias
        if use_rel_pos_bias:
            self.rel_pos_bias = RelPosBias(n_heads, max_seq_len)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        """Initialize model parameters."""
        for name, param in self.named_parameters():
            if 'weight' in name and len(param.shape) > 1:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)

    def _time_diff_to_bucket(self, time_diffs):
        """Map time differences to bucket indices."""
        time_diffs = time_diffs.float() / 60.0  # seconds to minutes
        time_diffs = torch.clamp(time_diffs, min=1e-6)

        if self.time_bucket_fn == 'sqrt':
            buckets = torch.sqrt(time_diffs).long()
        elif self.time_bucket_fn == 'log':
            buckets = torch.log(time_diffs).long()
        else:
            raise ValueError(f"Unsupported time_bucket_fn: {self.time_bucket_fn}")

        buckets = torch.clamp(buckets, min=0, max=self.num_time_buckets - 1)
        return buckets

    def forward(self, seq_tokens, time_diffs=None):
        """Forward pass.
        
        Args:
            seq_tokens (Tensor): Item token IDs of shape (B, L).
            time_diffs (Tensor, optional): Time differences in seconds of shape (B, L).
            
        Returns:
            Tensor: Logits of shape (B, L, vocab_size).
        """
        batch_size, seq_len = seq_tokens.shape

        # Look up item embeddings
        item_emb = self.item_embeddings[seq_tokens]  # (B, L, D)

        # Add positional embedding
        positions = torch.arange(seq_len, dtype=torch.long, device=seq_tokens.device)
        pos_emb = self.position_embedding(positions)  # (L, D)
        embeddings = item_emb + pos_emb.unsqueeze(0)  # (B, L, D)

        # Add time embedding if provided
        if self.use_time_embedding:
            if time_diffs is None:
                time_diffs = torch.zeros(batch_size, seq_len, dtype=torch.long, device=seq_tokens.device)

            time_buckets = self._time_diff_to_bucket(time_diffs)
            time_emb = self.time_embedding(time_buckets)  # (B, L, D)
            embeddings = embeddings + time_emb

        embeddings = self.dropout(embeddings)

        # Get relative position bias
        rel_pos_bias = None
        if self.use_rel_pos_bias:
            rel_pos_bias = self.rel_pos_bias(seq_len)

        # Pass through transformer blocks
        x = embeddings
        for block in self.transformer_blocks:
            x = block(x, rel_pos_bias=rel_pos_bias)

        # Scoring head: compute dot product with item embeddings
        # x: (B, L, D), item_embeddings: (V, D)
        logits = torch.matmul(x, self.item_embeddings.t()) / self.temperature  # (B, L, V)

        return logits
