"""HSTU: Hierarchical Sequential Transduction Units Model."""

import math

import torch
import torch.nn as nn

from torch_rechub.basic.layers import HSTUBlock
from torch_rechub.utils.hstu_utils import RelPosBias


class HSTUModel(nn.Module):
    """HSTU: Hierarchical Sequential Transduction Units.

    Autoregressive generative recommender that stacks ``HSTUBlock`` layers to
    capture long-range dependencies and predict the next item.

    Parameters
    ----------
    vocab_size : int
        Vocabulary size (items incl. PAD).
    d_model : int, default=512
        Hidden dimension.
    n_heads : int, default=8
        Attention heads.
    n_layers : int, default=4
        Number of stacked HSTU layers.
    dqk : int, default=64
        Query/key dim per head.
    dv : int, default=64
        Value dim per head.
    max_seq_len : int, default=256
        Maximum sequence length.
    dropout : float, default=0.1
        Dropout rate.
    use_rel_pos_bias : bool, default=True
        Use relative position bias.
    use_time_embedding : bool, default=True
        Use time-difference embeddings.
    num_time_buckets : int, default=2048
        Number of time buckets for time embeddings.
    time_bucket_fn : {'sqrt', 'log'}, default='sqrt'
        Bucketization function for time differences.

    Shape
    -----
    Input
        x : ``(batch_size, seq_len)``
        time_diffs : ``(batch_size, seq_len)``, optional (seconds).
    Output
        logits : ``(batch_size, seq_len, vocab_size)``

    Examples
    --------
    >>> model = HSTUModel(vocab_size=100000, d_model=512)
    >>> x = torch.randint(0, 100000, (32, 256))
    >>> time_diffs = torch.randint(0, 86400, (32, 256))
    >>> logits = model(x, time_diffs)
    >>> logits.shape
    torch.Size([32, 256, 100000])
    """

    def __init__(self, vocab_size, d_model=512, n_heads=8, n_layers=4, dqk=64, dv=64, max_seq_len=256, dropout=0.1, use_rel_pos_bias=True, use_time_embedding=True, num_time_buckets=2048, time_bucket_fn='sqrt'):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        self.use_time_embedding = use_time_embedding
        self.num_time_buckets = num_time_buckets
        self.time_bucket_fn = time_bucket_fn

        # Alpha scaling factor (following the Meta reference implementation)
        self.alpha = math.sqrt(d_model)

        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)

        # Absolute positional embedding
        self.position_embedding = nn.Embedding(max_seq_len, d_model)

        # Time embedding
        if use_time_embedding:
            # Embedding table for time-difference buckets
            # num_time_buckets + 1: extra bucket reserved for padding
            self.time_embedding = nn.Embedding(num_time_buckets + 1, d_model, padding_idx=0)

        # HSTU block
        self.hstu_block = HSTUBlock(d_model=d_model, n_heads=n_heads, n_layers=n_layers, dqk=dqk, dv=dv, dropout=dropout, use_rel_pos_bias=use_rel_pos_bias)

        # Relative position bias
        self.use_rel_pos_bias = use_rel_pos_bias
        if use_rel_pos_bias:
            self.rel_pos_bias = RelPosBias(n_heads, max_seq_len)

        # 输出投影层
        self.output_projection = nn.Linear(d_model, vocab_size)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Initialize model parameters
        self._init_weights()

    def _init_weights(self):
        """Initialize model parameters."""
        for name, param in self.named_parameters():
            if 'weight' in name and len(param.shape) > 1:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)

    def _time_diff_to_bucket(self, time_diffs):
        """Map raw time differences (in seconds) to discrete bucket indices.

        Following the Meta HSTU implementation, continuous time differences are
        first converted to minutes and then bucketized using either a square-root
        or logarithmic transform.

        Args:
            time_diffs (Tensor): Time differences in seconds,
                shape ``(batch_size, seq_len)``.

        Returns:
            Tensor: Integer bucket indices of shape ``(batch_size, seq_len)``.
        """
        # Convert seconds to minutes (as in the Meta reference implementation)
        time_bucket_increments = 60.0
        time_diffs = time_diffs.float() / time_bucket_increments

        # Ensure non-negative values and avoid log(0)
        time_diffs = torch.clamp(time_diffs, min=1e-6)

        if self.time_bucket_fn == 'sqrt':
            # Use the square-root transform: suitable when time differences
            # are relatively evenly distributed.
            buckets = torch.sqrt(time_diffs).long()
        elif self.time_bucket_fn == 'log':
            # Use the logarithmic transform: suitable when time differences
            # span several orders of magnitude.
            buckets = torch.log(time_diffs).long()
        else:
            raise ValueError(f"Unsupported time_bucket_fn: {self.time_bucket_fn}")

        # Clamp bucket indices to the valid range [0, num_time_buckets - 1]
        buckets = torch.clamp(buckets, min=0, max=self.num_time_buckets - 1)

        return buckets

    def forward(self, x, time_diffs=None):
        """Forward pass.

        Args:
            x (Tensor): Input token ids of shape ``(batch_size, seq_len)``.
            time_diffs (Tensor, optional): Time differences in seconds,
                shape ``(batch_size, seq_len)``. If ``None`` and
                ``use_time_embedding=True``, all-zero time differences are used.

        Returns:
            Tensor: Logits over the vocabulary of shape
                ``(batch_size, seq_len, vocab_size)``.
        """
        batch_size, seq_len = x.shape

        # Token embedding with alpha scaling (as in the Meta implementation)
        token_emb = self.token_embedding(x) * self.alpha  # (B, L, D)

        # Absolute positional embedding
        positions = torch.arange(seq_len, dtype=torch.long, device=x.device)
        pos_emb = self.position_embedding(positions)  # (L, D)

        # Combine token and position embeddings
        embeddings = token_emb + pos_emb.unsqueeze(0)  # (B, L, D)

        # Optional time-difference embedding
        if self.use_time_embedding:
            if time_diffs is None:
                # Fallback: use all-zero time differences when none are provided
                time_diffs = torch.zeros(batch_size, seq_len, dtype=torch.long, device=x.device)

            # Map raw time differences to bucket indices
            time_buckets = self._time_diff_to_bucket(time_diffs)  # (B, L)

            # Look up time embeddings and add to the sequence representation
            time_emb = self.time_embedding(time_buckets)  # (B, L, D)

            # embeddings = token_emb + pos_emb + time_emb
            embeddings = embeddings + time_emb

        embeddings = self.dropout(embeddings)

        # Relative position bias for self-attention
        rel_pos_bias = None
        if self.use_rel_pos_bias:
            rel_pos_bias = self.rel_pos_bias(seq_len)  # (1, H, L, L)

        # HSTU block
        hstu_output = self.hstu_block(embeddings, rel_pos_bias=rel_pos_bias)  # (B, L, D)

        # Final projection to vocabulary logits
        logits = self.output_projection(hstu_output)  # (B, L, V)

        return logits
