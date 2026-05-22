"""Utility classes for HSTU / HLLM."""

import math

import torch
import torch.nn as nn


class RelPosBias(nn.Module):
    """Legacy relative-position bias for attention.

    Used by ``HLLMTransformerBlock`` and kept for backward-compatible
    experiments. ``HSTUModel`` uses
    :class:`RelativeBucketedTimeAndPositionBias` instead, because HSTU Eq. 3
    adds a per-head bucketed ``rab^{p,t}`` term directly to attention scores.

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
        """Compute legacy relative-position bias.

        Args:
            seq_len (int): Sequence length ``L``.

        Returns:
            Tensor: Relative-position bias with shape ``(1, n_heads, L, L)``.
        """
        positions = torch.arange(seq_len, dtype=torch.long, device=self.rel_pos_bias_table.device)
        relative_positions = positions.unsqueeze(0) - positions.unsqueeze(1)
        buckets = self._relative_position_bucket(relative_positions)
        bias = self.rel_pos_bias_table[buckets]  # (L, L, n_heads)
        bias = bias.permute(2, 0, 1).unsqueeze(0)  # (1, n_heads, L, L)
        return bias


class RelativeBucketedTimeAndPositionBias(nn.Module):
    """HSTU ``rab^{p,t}``: per-head bias on attention scores from (position-diff,
    time-diff) pairs, following the HSTU paper (Eq. 3) and Meta's reference
    ``RelativeBucketedTimeAndPositionBasedBias``.

    The bias is added to ``Q K^T`` **before** the ``SiLU/N`` activation.

    Parameters
    ----------
    n_heads : int
        Number of attention heads (per-head bias).
    max_seq_len : int
        Maximum sequence length. Sizes the position table to
        ``2 * max_seq_len - 1`` slots.
    num_time_buckets : int, default=128
        Number of time-difference buckets; an extra OOB slot is appended so the
        bucket index range is ``[0, num_time_buckets]`` inclusive.
    time_bucket_fn : {'sqrt', 'log'}, default='sqrt'
        Bucketization function applied to ``|dt|`` in minutes.
    time_bucket_divisor : float, default=1.0
        Divisor applied after ``sqrt``/``log`` so the bucket index range
        actually utilizes ``[0, num_time_buckets]``.
    """

    def __init__(self, n_heads, max_seq_len, num_time_buckets=128, time_bucket_fn='sqrt', time_bucket_divisor=1.0):
        super().__init__()
        if time_bucket_fn not in ('sqrt', 'log'):
            raise ValueError(f"Unsupported time_bucket_fn: {time_bucket_fn}")
        self.n_heads = n_heads
        self.max_seq_len = max_seq_len
        self.num_time_buckets = num_time_buckets
        self.time_bucket_fn = time_bucket_fn
        self.time_bucket_divisor = time_bucket_divisor

        bound_pos = math.sqrt(1.0 / (2 * max_seq_len - 1))
        self.pos_w = nn.Parameter(torch.empty(2 * max_seq_len - 1, n_heads).uniform_(-bound_pos, bound_pos))
        bound_ts = math.sqrt(1.0 / (num_time_buckets + 1))
        self.ts_w = nn.Parameter(torch.empty(num_time_buckets + 1, n_heads).uniform_(-bound_ts, bound_ts))

    def _bucketize_time(self, dt):
        """Map signed seconds deltas to bucket indices in ``[0, num_time_buckets]``."""
        dt = dt.float().abs() / 60.0
        dt = torch.clamp(dt, min=1e-6)
        if self.time_bucket_fn == 'sqrt':
            buckets = torch.sqrt(dt)
        else:
            buckets = torch.log(dt)
        return (buckets / self.time_bucket_divisor).clamp(min=0, max=self.num_time_buckets).long()

    def forward(self, time_diffs=None, seq_len=None):
        """Return relative bias for adding to attention scores.

        ``time_diffs`` follows the preprocessing convention where each entry is
        ``anchor - timestamp[i]`` (or 0 at PAD). The pairwise difference
        ``time_diffs[i] - time_diffs[j]`` recovers ``ts[j] - ts[i]``, so the
        anchor cancels.

        Returns
        -------
        Tensor
            ``(B, H, L, L)`` when ``time_diffs`` is given;
            ``(1, H, L, L)`` (position-only) when ``time_diffs`` is ``None``.
        """
        if time_diffs is None:
            if seq_len is None:
                raise ValueError("Provide either `time_diffs` or `seq_len`.")
            L = seq_len
            device = self.pos_w.device
        else:
            _, L = time_diffs.shape
            device = time_diffs.device

        if L > self.max_seq_len:
            raise ValueError(f"seq_len ({L}) exceeds max_seq_len ({self.max_seq_len}).")

        positions = torch.arange(L, device=device)
        # rel = i - j in [-(L-1), L-1]; offset into pos_w's [0, 2*max_seq_len-2] range.
        rel_pos_idx = positions.unsqueeze(0) - positions.unsqueeze(1) + (self.max_seq_len - 1)
        pos_bias = self.pos_w[rel_pos_idx].permute(2, 0, 1)  # (H, L, L)

        if time_diffs is None:
            return pos_bias.unsqueeze(0)  # (1, H, L, L)

        dt_pairwise = time_diffs.unsqueeze(2) - time_diffs.unsqueeze(1)  # (B, L, L)
        time_bias = self.ts_w[self._bucketize_time(dt_pairwise)]  # (B, L, L, H)
        time_bias = time_bias.permute(0, 3, 1, 2)  # (B, H, L, L)
        return pos_bias.unsqueeze(0) + time_bias  # (B, H, L, L)


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
            torch.ones(vocab_size,
                       dtype=torch.bool),
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
