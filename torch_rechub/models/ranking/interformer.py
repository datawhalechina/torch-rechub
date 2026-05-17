"""
Date: create on 05/04/2026
References:
    paper: InterFormer: Effective Heterogeneous Interaction Learning for Click-Through Rate Prediction
    url: https://arxiv.org/abs/2411.09852
Authors: Implemented based on the paper by Zeng et al. (Meta AI, 2025)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...basic.layers import MLP, CrossNetV2, CrossNetMix, EmbeddingLayer, InputMask


class Gating(nn.Module):
    """Self-gating module for selective information aggregation.

    Implements the gating mechanism: Gating(X) = σ(X ⊙ MLP(X))

    Args:
        input_dim (int): Input feature dimension.
        hidden_dim (int, optional): Hidden dimension for the gating MLP.
            If None, uses input_dim.

    Shape:
        - Input: (batch_size, num_features, embed_dim)
        - Output: (batch_size, num_features, embed_dim)
    """

    def __init__(self, input_dim, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or input_dim
        self.gate_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, input_dim),
        )
        self.activation = nn.Sigmoid()

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, num_features, embed_dim)
        Returns:
            Gated tensor of shape (batch_size, num_features, embed_dim)
        """
        gate = self.activation(x * self.gate_mlp(x))
        return gate * x


class PFFN(nn.Module):
    """Personalized FeedForward Network.

    Learns transformation weights based on non-sequence summarization
    and applies them to sequence embeddings.

    PFFN(X_sum, S) = f(X_sum) * S, where f is an MLP.

    Args:
        input_dim (int): Dimension of the non-sequence summarization.
        seq_dim (int): Dimension of sequence embeddings.
        hidden_dim (int, optional): Hidden dimension for the weight MLP.

    Shape:
        - X_sum: (batch_size, num_sum_features, embed_dim)
        - S: (batch_size, seq_len, embed_dim)
        - Output: (batch_size, seq_len, embed_dim)
    """

    def __init__(self, input_dim, seq_dim, hidden_dim=None):
        super().__init__()
        self.input_dim = input_dim
        self.seq_dim = seq_dim
        hidden_dim = hidden_dim or input_dim

        # Learn transformation weight based on non-sequence summarization
        # Output: (batch_size, num_sum_features, seq_dim)
        self.weight_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, seq_dim),
        )

    def forward(self, x_sum, seq):
        """
        Args:
            x_sum: Non-sequence summarization of shape (batch_size, num_sum_features, embed_dim)
            seq: Sequence embeddings of shape (batch_size, seq_len, embed_dim)
        Returns:
            Transformed sequence embeddings of shape (batch_size, seq_len, embed_dim)
        """
        # Compute transformation weight: (batch_size, num_sum_features, seq_dim)
        weight = self.weight_mlp(x_sum)
        # Aggregate across sum features: (batch_size, seq_dim)
        weight = weight.mean(dim=1)
        # Apply transformation: (batch_size, seq_len, embed_dim)
        # weight.unsqueeze(1) broadcasts across seq_len
        output = weight.unsqueeze(1) * seq
        return output


class PoolingByMultiHeadAttention(nn.Module):
    """Pooling by Multi-Head Attention (PMA).

    Uses learnable query vectors to summarize the sequence from k different aspects.

    Args:
        embed_dim (int): Embedding dimension.
        num_heads (int): Number of attention heads.
        num_seeds (int): Number of learnable seed vectors (k in the paper).
        dropout (float): Dropout rate.

    Shape:
        - Input: (batch_size, seq_len, embed_dim)
        - Output: (batch_size, num_seeds, embed_dim)
    """

    def __init__(self, embed_dim, num_heads=4, num_seeds=2, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_seeds = num_seeds
        self.head_dim = embed_dim // num_heads

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        # Learnable seed vectors (queries)
        self.seed = nn.Parameter(torch.randn(1, num_seeds, embed_dim))

        # Projections for multi-head attention
        self.W_Q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_K = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_V = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_O = nn.Linear(embed_dim, embed_dim, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(self, seq):
        """
        Args:
            seq: Sequence embeddings of shape (batch_size, seq_len, embed_dim)
        Returns:
            Summarized embeddings of shape (batch_size, num_seeds, embed_dim)
        """
        batch_size = seq.size(0)

        # Expand seed to batch size: (batch_size, num_seeds, embed_dim)
        Q = self.seed.expand(batch_size, -1, -1)
        K = seq  # (batch_size, seq_len, embed_dim)
        V = seq  # (batch_size, seq_len, embed_dim)

        # Linear projections
        Q = self.W_Q(Q)  # (batch_size, num_seeds, embed_dim)
        K = self.W_K(K)  # (batch_size, seq_len, embed_dim)
        V = self.W_V(V)  # (batch_size, seq_len, embed_dim)

        # Reshape for multi-head attention
        # Q: (batch_size, num_heads, num_seeds, head_dim)
        # K, V: (batch_size, num_heads, seq_len, head_dim)
        Q = Q.view(batch_size, self.num_seeds, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)  # (batch_size, num_heads, num_seeds, head_dim)

        # Reshape back: (batch_size, num_seeds, embed_dim)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, self.num_seeds, self.embed_dim)

        # Output projection
        output = self.W_O(attn_output)

        return output


class MaskNet(nn.Module):
    """MaskNet for multi-sequence unification and denoising.

    Unifies multiple sequences and filters out internal noises via self-masking.

    Args:
        input_dim (int): Total dimension of concatenated sequences (k * embed_dim).
        output_dim (int): Output dimension (embed_dim).
        hidden_dim (int, optional): Hidden dimension for mask MLP.

    Shape:
        - Input: (batch_size, seq_len, input_dim)
        - Output: (batch_size, seq_len, output_dim)
    """

    def __init__(self, input_dim, output_dim, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or input_dim // 2

        # Self-masking MLP
        self.mask_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid(),
        )

        # Linear combination MLP (LCE in paper)
        self.lce_mlp = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        """
        Args:
            x: Concatenated sequences of shape (batch_size, seq_len, input_dim)
        Returns:
            Unified sequence of shape (batch_size, seq_len, output_dim)
        """
        # Compute self-masking
        mask = self.mask_mlp(x)
        # Apply mask
        masked_x = x * mask
        # Linear combination
        output = self.lce_mlp(masked_x)
        return output


class RotaryPositionalEncoding(nn.Module):
    """Rotary Position Embedding (RoPE).

    Implements rotary position embeddings for better position awareness.

    Args:
        embed_dim (int): Embedding dimension.
        max_seq_len (int): Maximum sequence length.

    Shape:
        - Input: (batch_size, seq_len, embed_dim)
        - Output: (batch_size, seq_len, embed_dim)
    """

    def __init__(self, embed_dim, max_seq_len=512):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len

        # Compute inverse frequencies
        inv_freq = 1.0 / (10000 ** (torch.arange(0, embed_dim, 2).float() / embed_dim))
        self.register_buffer("inv_freq", inv_freq)

        # Precompute position encodings
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len):
        """Precompute position encodings."""
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # (seq_len, embed_dim/2) -> (seq_len, embed_dim)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)
        Returns:
            Tensor with rotary position embeddings applied
        """
        batch_size, seq_len, embed_dim = x.shape

        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)

        # Get position embeddings
        cos = self.cos_cached[:seq_len]  # (seq_len, embed_dim)
        sin = self.sin_cached[:seq_len]  # (seq_len, embed_dim)

        # Apply rotary embedding using the rotation formula
        # rotate_half: [x1, x2] -> [-x2, x1]
        x1 = x[..., :embed_dim // 2]
        x2 = x[..., embed_dim // 2:]
        rotated = torch.cat([-x2, x1], dim=-1)

        # Apply rotation
        x_out = x * cos.unsqueeze(0) + rotated * sin.unsqueeze(0)

        return x_out


class MultiHeadAttention(nn.Module):
    """Multi-Head Self-Attention for Sequence Arch.

    Args:
        embed_dim (int): Embedding dimension.
        num_heads (int): Number of attention heads.
        dropout (float): Dropout rate.

    Shape:
        - Input: (batch_size, seq_len, embed_dim)
        - Output: (batch_size, seq_len, embed_dim)
    """

    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.W_Q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_K = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_V = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_O = nn.Linear(embed_dim, embed_dim, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(self, x, mask=None):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)
            mask: Optional mask of shape (batch_size, seq_len)
        Returns:
            Output tensor of shape (batch_size, seq_len, embed_dim)
        """
        batch_size, seq_len, _ = x.shape

        # Linear projections
        Q = self.W_Q(x)  # (batch_size, seq_len, embed_dim)
        K = self.W_K(x)
        V = self.W_V(x)

        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        # Apply mask if provided
        if mask is not None:
            # mask: (batch_size, seq_len) -> (batch_size, 1, 1, seq_len)
            mask = mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)  # (batch_size, num_heads, seq_len, head_dim)

        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)

        # Output projection
        output = self.W_O(attn_output)

        return output


class InteractionArch(nn.Module):
    """Interaction Arch for learning behavior-aware non-sequence embeddings.

    Models interactions among non-sequence features and sequence summarization.

    Args:
        input_dim (int): Total input dimension (non-seq dim + seq_sum dim).
        output_dim (int): Output dimension.
        interaction_type (str): Type of interaction module ('dot', 'dcnv2', 'dhen').
        n_cross_layers (int): Number of cross layers for DCNv2.
        use_low_rank (bool): Whether to use low-rank mixture for DCNv2.
        low_rank (int): Low-rank dimension for DCNv2.
        num_experts (int): Number of experts for DCNv2 mixture.

    Shape:
        - X: (batch_size, num_non_seq_features, embed_dim)
        - S_sum: (batch_size, num_seq_sum_features, embed_dim)
        - Output: (batch_size, num_non_seq_features, embed_dim)
    """

    def __init__(self, input_dim, output_dim, interaction_type="dcnv2",
                 n_cross_layers=2, use_low_rank=True, low_rank=32, num_experts=4):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.interaction_type = interaction_type

        if interaction_type == "dcnv2":
            if use_low_rank:
                self.interaction = CrossNetMix(input_dim, n_cross_layers, low_rank, num_experts)
            else:
                self.interaction = CrossNetV2(input_dim, n_cross_layers)
        elif interaction_type == "dot":
            # Simple dot product interaction
            self.interaction = None
        else:
            raise ValueError(f"Unsupported interaction type: {interaction_type}")

        # MLP to transform output back to match input shape
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.SiLU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, X, S_sum):
        """
        Args:
            X: Non-sequence embeddings of shape (batch_size, num_features, embed_dim)
            S_sum: Sequence summarization of shape (batch_size, num_sum_features, embed_dim)
        Returns:
            Updated non-sequence embeddings of shape (batch_size, num_features, embed_dim)
        """
        batch_size, num_features, embed_dim = X.shape

        # Concatenate X and S_sum
        combined = torch.cat([X, S_sum], dim=1)  # (batch_size, num_features + num_sum, embed_dim)
        combined_flat = combined.flatten(start_dim=1)  # (batch_size, (num_features + num_sum) * embed_dim)

        # Apply interaction
        if self.interaction_type == "dcnv2":
            interacted = self.interaction(combined_flat)
        else:  # dot product
            # Simple dot product interaction
            interacted = combined_flat

        # Transform back to feature shape
        output = self.mlp(interacted)  # (batch_size, output_dim)
        output = output.view(batch_size, -1, embed_dim)  # (batch_size, num_features, embed_dim)

        return output


class SequenceArch(nn.Module):
    """Sequence Arch for learning context-aware sequence embeddings.

    Uses PFFN and Multi-Head Attention for sequence modeling.

    Args:
        embed_dim (int): Embedding dimension.
        num_heads (int): Number of attention heads.
        dropout (float): Dropout rate.
        use_rope (bool): Whether to use rotary position embeddings.
        max_seq_len (int): Maximum sequence length for RoPE.

    Shape:
        - S: (batch_size, seq_len, embed_dim)
        - X_sum: (batch_size, num_sum_features, embed_dim)
        - Output: (batch_size, seq_len, embed_dim)
    """

    def __init__(self, embed_dim, num_heads=4, dropout=0.1, use_rope=True, max_seq_len=512):
        super().__init__()
        self.embed_dim = embed_dim

        # Personalized FFN
        self.pffn = PFFN(embed_dim, embed_dim)

        # Multi-Head Attention
        self.mha = MultiHeadAttention(embed_dim, num_heads, dropout)

        # Rotary Position Encoding
        self.use_rope = use_rope
        if use_rope:
            self.rope = RotaryPositionalEncoding(embed_dim, max_seq_len)

        # Layer Normalization
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, S, X_sum, mask=None):
        """
        Args:
            S: Sequence embeddings of shape (batch_size, seq_len, embed_dim)
            X_sum: Non-sequence summarization of shape (batch_size, num_sum, embed_dim)
            mask: Optional mask of shape (batch_size, seq_len)
        Returns:
            Updated sequence embeddings of shape (batch_size, seq_len, embed_dim)
        """
        # Apply PFFN
        S_pffn = self.pffn(X_sum, S)

        # Apply rotary position encoding
        if self.use_rope:
            S_pffn = self.rope(S_pffn)

        # Multi-Head Attention with residual
        S_attn = self.norm1(S_pffn)
        S_attn = self.mha(S_attn, mask)
        S = S + S_attn

        # FFN with residual
        S_ffn = self.norm2(S)
        S_ffn = self.ffn(S_ffn)
        S = S + S_ffn

        return S


class CrossArch(nn.Module):
    """Cross Arch for effective information selection and summarization.

    Selects and summarizes information before exchanging between
    Interaction and Sequence Arch.

    Args:
        embed_dim (int): Embedding dimension.
        num_non_seq_features (int): Number of non-sequence features.
        num_cls_tokens (int): Number of CLS tokens.
        num_pma_seeds (int): Number of PMA seeds.
        num_recent_tokens (int): Number of recent tokens to keep.
        num_heads (int): Number of attention heads for PMA.
        num_sum_features (int): Number of summarized features for non-seq.

    Shape:
        - X: (batch_size, num_non_seq_features, embed_dim)
        - S: (batch_size, seq_len, embed_dim)
        - Output X_sum: (batch_size, num_sum_features, embed_dim)
        - Output S_sum: (batch_size, num_sum_features, embed_dim)
    """

    def __init__(self, embed_dim, num_non_seq_features, num_cls_tokens=4, num_pma_seeds=2,
                 num_recent_tokens=2, num_heads=4, num_sum_features=8):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_cls_tokens = num_cls_tokens
        self.num_pma_seeds = num_pma_seeds
        self.num_recent_tokens = num_recent_tokens
        self.num_sum_features = num_sum_features

        # Gating for non-sequence summarization
        self.non_seq_gating = Gating(embed_dim * num_sum_features)

        # PMA for sequence summarization
        self.pma = PoolingByMultiHeadAttention(embed_dim, num_heads, num_pma_seeds)

        # Gating for sequence summarization
        total_seq_sum_dim = (num_cls_tokens + num_pma_seeds + num_recent_tokens) * embed_dim
        self.seq_gating = Gating(total_seq_sum_dim)

        # Linear projection to compress non-seq features
        # Input: num_non_seq_features * embed_dim, Output: num_sum_features * embed_dim
        self.compress_mlp = nn.Sequential(
            nn.Linear(num_non_seq_features * embed_dim, num_sum_features * embed_dim),
            nn.SiLU(),
        )

    def forward(self, X, S, S_cls=None):
        """
        Args:
            X: Non-sequence embeddings of shape (batch_size, num_features, embed_dim)
            S: Sequence embeddings of shape (batch_size, seq_len, embed_dim)
            S_cls: CLS tokens from previous layer of shape (batch_size, num_cls, embed_dim)
        Returns:
            X_sum: Non-sequence summarization of shape (batch_size, num_sum, embed_dim)
            S_sum: Sequence summarization of shape (batch_size, num_sum, embed_dim)
            S_cls: CLS tokens for next layer
        """
        batch_size = X.size(0)

        # Non-sequence summarization with gating
        X_flat = X.flatten(start_dim=1)  # (batch_size, num_features * embed_dim)
        X_compressed = self.compress_mlp(X_flat)  # (batch_size, num_sum * embed_dim)
        X_sum = self.non_seq_gating(X_compressed)
        X_sum = X_sum.view(batch_size, self.num_sum_features, self.embed_dim)

        # Sequence summarization
        # 1. CLS tokens (first num_cls_tokens from sequence after attention)
        if S_cls is not None:
            cls_tokens = S[:, :self.num_cls_tokens, :]
        else:
            cls_tokens = S[:, :self.num_cls_tokens, :]

        # 2. PMA tokens
        pma_tokens = self.pma(S)  # (batch_size, num_pma_seeds, embed_dim)

        # 3. Recent tokens
        recent_tokens = S[:, -self.num_recent_tokens:, :]  # (batch_size, num_recent, embed_dim)

        # Concatenate and gate
        S_sum_concat = torch.cat([cls_tokens, pma_tokens, recent_tokens], dim=1)
        S_sum_flat = S_sum_concat.flatten(start_dim=1)
        S_sum_flat = self.seq_gating(S_sum_flat)
        S_sum = S_sum_flat.view(batch_size, -1, self.embed_dim)

        return X_sum, S_sum, cls_tokens


class InterFormer(nn.Module):
    """InterFormer: Effective Heterogeneous Interaction Learning for CTR Prediction.

    This model enables bidirectional information flow between non-sequence
    and sequence features through an interleaving learning style.

    Args:
        features (list): Non-sequence features (SparseFeature and DenseFeature).
        history_features (list): Sequence features (SequenceFeature).
        target_features (list): Target item features.
        mlp_params (dict): Parameters for the final MLP classifier.
        embed_dim (int): Embedding dimension for all features.
        n_layers (int): Number of InterFormer layers.
        num_heads (int): Number of attention heads.
        dropout (float): Dropout rate.
        num_cls_tokens (int): Number of CLS tokens.
        num_pma_seeds (int): Number of PMA seeds.
        num_recent_tokens (int): Number of recent tokens for summarization.
        num_sum_features (int): Number of summarized features.
        interaction_type (str): Type of interaction module ('dot', 'dcnv2').
        n_cross_layers (int): Number of cross layers for DCNv2.
        use_low_rank (bool): Whether to use low-rank mixture for DCNv2.
        low_rank (int): Low-rank dimension for DCNv2.
        use_rope (bool): Whether to use rotary position embeddings.
        max_seq_len (int): Maximum sequence length.

    Example:
        >>> from torch_rechub.basic.features import SparseFeature, SequenceFeature
        >>> user_feat = SparseFeature("user_id", vocab_size=10000, embed_dim=64)
        >>> item_feat = SparseFeature("item_id", vocab_size=50000, embed_dim=64)
        >>> history_feat = SequenceFeature("history_item_ids", vocab_size=50000, 
        ...                                 embed_dim=64, pooling="concat")
        >>> model = InterFormer(
        ...     features=[user_feat],
        ...     history_features=[history_feat],
        ...     target_features=[item_feat],
        ...     mlp_params={"dims": [256, 128]},
        ...     embed_dim=64,
        ...     n_layers=2,
        ... )
        >>> # x: dict with feature tensors
        >>> y = model(x)
    """

    def __init__(self, features, history_features, target_features, mlp_params,
                 embed_dim=64, n_layers=2, num_heads=4, dropout=0.1,
                 num_cls_tokens=4, num_pma_seeds=2, num_recent_tokens=2,
                 num_sum_features=8, interaction_type="dcnv2", n_cross_layers=2,
                 use_low_rank=True, low_rank=32, use_rope=True, max_seq_len=512):
        super().__init__()
        self.features = features
        self.history_features = history_features
        self.target_features = target_features
        self.embed_dim = embed_dim
        self.n_layers = n_layers
        self.num_cls_tokens = num_cls_tokens
        self.num_sum_features = num_sum_features

        self.num_history_features = len(history_features)
        self.num_non_seq_features = len(features) + len(target_features)

        # Embedding layer for all features
        self.embedding = EmbeddingLayer(features + history_features + target_features)
        self.input_mask = InputMask()

        # MaskNet for multi-sequence unification
        if self.num_history_features > 1:
            self.masknet = MaskNet(
                input_dim=embed_dim * self.num_history_features,
                output_dim=embed_dim
            )
        else:
            self.masknet = None

        # Input dimension for Interaction Arch
        # Non-seq features + sequence summarization
        total_non_seq = self.num_non_seq_features + num_sum_features
        interaction_input_dim = total_non_seq * embed_dim
        interaction_output_dim = self.num_non_seq_features * embed_dim

        # Stack of InterFormer layers
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            layer = nn.ModuleDict({
                "cross_arch": CrossArch(
                    embed_dim=embed_dim,
                    num_non_seq_features=self.num_non_seq_features,
                    num_cls_tokens=num_cls_tokens,
                    num_pma_seeds=num_pma_seeds,
                    num_recent_tokens=num_recent_tokens,
                    num_heads=num_heads,
                    num_sum_features=num_sum_features,
                ),
                "interaction_arch": InteractionArch(
                    input_dim=interaction_input_dim,
                    output_dim=interaction_output_dim,
                    interaction_type=interaction_type,
                    n_cross_layers=n_cross_layers,
                    use_low_rank=use_low_rank,
                    low_rank=low_rank,
                ),
                "sequence_arch": SequenceArch(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    use_rope=use_rope,
                    max_seq_len=max_seq_len,
                ),
            })
            self.layers.append(layer)

        # Final MLP classifier
        # Input: non-seq sum + CLS tokens
        final_dim = (num_sum_features + num_cls_tokens) * embed_dim
        self.mlp = MLP(final_dim, **mlp_params)

    def forward(self, x):
        """
        Args:
            x: Dict containing feature tensors
                - Non-seq features: (batch_size,)
                - History features: (batch_size, seq_len)
                - Target features: (batch_size,)
        Returns:
            CTR prediction of shape (batch_size,)
        """
        # Embed features
        # (batch_size, num_features, embed_dim)
        embed_x_features = self.embedding(x, self.features)
        # (batch_size, num_target_features, embed_dim)
        embed_x_target = self.embedding(x, self.target_features)
        # (batch_size, num_history_features, seq_len, embed_dim)
        embed_x_history = self.embedding(x, self.history_features)

        # Get sequence mask
        history_mask = self.input_mask(x, self.history_features)  # (batch_size, num_history, seq_len)

        # Process sequence features
        batch_size = embed_x_features.size(0)
        seq_len = embed_x_history.size(2)

        # Combine multiple history sequences
        if self.masknet is not None and self.num_history_features > 1:
            # (batch_size, num_history, seq_len, embed_dim) -> (batch_size, seq_len, num_history * embed_dim)
            history_flat = embed_x_history.permute(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
            S = self.masknet(history_flat)  # (batch_size, seq_len, embed_dim)
            # Update mask
            mask = history_mask[:, 0, :]  # Use first history mask
        elif self.num_history_features == 1:
            S = embed_x_history[:, 0, :, :]  # (batch_size, seq_len, embed_dim)
            mask = history_mask[:, 0, :]
        else:
            raise ValueError("At least one history feature is required")

        # Combine non-seq features and target features
        X = torch.cat([embed_x_features, embed_x_target], dim=1)  # (batch_size, num_non_seq, embed_dim)

        # Initialize CLS tokens (using non-seq summarization)
        X_sum_init = X.mean(dim=1, keepdim=True).expand(-1, self.num_cls_tokens, -1)
        S = torch.cat([X_sum_init, S], dim=1)  # Prepend CLS tokens
        # Update mask for CLS tokens
        cls_mask = torch.ones(batch_size, self.num_cls_tokens, device=mask.device)
        mask = torch.cat([cls_mask, mask], dim=1)

        # Store CLS tokens for first layer
        S_cls = None

        # Process through InterFormer layers
        for layer in self.layers:
            cross_arch = layer["cross_arch"]
            interaction_arch = layer["interaction_arch"]
            sequence_arch = layer["sequence_arch"]

            # Cross Arch: Get summarizations
            X_sum, S_sum, S_cls = cross_arch(X, S, S_cls)

            # Interaction Arch: Update non-seq embeddings
            X = interaction_arch(X, S_sum)

            # Sequence Arch: Update sequence embeddings
            S = sequence_arch(S, X_sum, mask)

        # Final prediction
        # Use CLS tokens from sequence and non-seq summarization
        final_cls = S[:, :self.num_cls_tokens, :]  # (batch_size, num_cls, embed_dim)
        X_sum_final = X.mean(dim=1, keepdim=True).expand(-1, self.num_sum_features, -1)

        # Concatenate for final MLP
        final_input = torch.cat([X_sum_final, final_cls], dim=1).flatten(start_dim=1)
        y = self.mlp(final_input)

        return torch.sigmoid(y.squeeze(1))
