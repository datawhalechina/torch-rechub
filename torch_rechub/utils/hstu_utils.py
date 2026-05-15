"""Utility classes for HSTU / HLLM."""

import math

import torch
import torch.nn as nn


class RelPosBias(nn.Module):
    """Relative position bias for attention.

    Used by ``HLLMTransformerBlock``. The HSTU reference does not use a
    relative bias on attention scores, so ``HSTUModel`` no longer wires this
    in; the class is kept here for HLLM and for opt-in HSTU experiments.

    Parameters
    ----------
    n_heads : int
        Number of attention heads.
    max_seq_len : int
        Maximum supported sequence length.
    num_buckets : int, default=32
        Number of relative position buckets.

    Shape
    -----
    Output: ``(1, n_heads, seq_len, seq_len)``
    """

    def __init__(self, n_heads, max_seq_len, num_buckets=32):
        super().__init__()
        self.n_heads = n_heads
        self.max_seq_len = max_seq_len
        self.num_buckets = num_buckets

        # Uniform init bounded by sqrt(1/num_buckets); avoids the large
        # additive bias on attention scores that `torch.randn` produces.
        bound = math.sqrt(1.0 / num_buckets)
        self.rel_pos_bias_table = nn.Parameter(torch.empty(num_buckets, n_heads).uniform_(-bound, bound))

    def _relative_position_bucket(self, relative_position):
        """Map relative positions to bucket indices in ``[0, num_buckets-1]``."""
        num_buckets = self.num_buckets
        max_distance = self.max_seq_len

        # Clamp to max_distance BEFORE bucketization so out-of-range positions
        # do not silently overflow into the last bucket via post-hoc clamp.
        relative_position = torch.abs(relative_position).clamp(max=max_distance)

        bucket = relative_position * (num_buckets - 1) // max_distance
        return bucket.long()

    def forward(self, seq_len):
        positions = torch.arange(seq_len, dtype=torch.long, device=self.rel_pos_bias_table.device)
        relative_positions = positions.unsqueeze(0) - positions.unsqueeze(1)
        buckets = self._relative_position_bucket(relative_positions)
        bias = self.rel_pos_bias_table[buckets]  # (L, L, n_heads)
        bias = bias.permute(2, 0, 1).unsqueeze(0)  # (1, n_heads, L, L)
        return bias


class VocabMask(nn.Module):
    """Vocabulary mask to block invalid items at inference / ranking.

    Parameters
    ----------
    vocab_size : int
        Vocabulary size.
    invalid_items : list, optional
        Token ids to mask out. ``[0]`` is the typical choice to drop PAD from
        next-item recommendations.
    """

    def __init__(self, vocab_size, invalid_items=None):
        super().__init__()
        self.vocab_size = vocab_size

        self.register_buffer(
            'mask',
            torch.ones(vocab_size, dtype=torch.bool),
        )

        if invalid_items is not None:
            for item_id in invalid_items:
                if 0 <= item_id < vocab_size:
                    self.mask[item_id] = False

    def apply_mask(self, logits):
        """Return ``logits`` with invalid item positions pushed to ``-1e9``."""
        masked_logits = logits.clone()
        masked_logits[..., ~self.mask] = -1e9
        return masked_logits
