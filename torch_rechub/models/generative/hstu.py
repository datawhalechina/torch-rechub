"""HSTU: Hierarchical Sequential Transduction Units Model."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_rechub.basic.layers import HSTUBlock


class HSTUModel(nn.Module):
    """HSTU: Hierarchical Sequential Transduction Units.

    Autoregressive generative recommender that stacks ``HSTUBlock`` layers to
    capture long-range dependencies and predict the next item. The layer
    internals follow the HSTU paper (Eq. 2-4) and Meta's reference
    implementation (`meta-recsys/generative-recommenders`):

    - Token embedding + absolute position embedding + (optional) time-bucket
      embedding on the input side.
    - Each HSTU layer applies a single ``SiLU`` to the joint ``UVQK``
      projection **before** splitting (Eq. 2), so all four streams go through
      the non-linearity.
    - Attention uses ``alpha = 1/sqrt(dqk)`` scaling and adds a per-head
      bucketed (position-diff, time-diff) bias ``rab^{p,t}`` to scores
      **before** the ``silu(scores) / max_seq_len`` activation (Eq. 3).
    - Gated output ``LayerNorm(A V) * U`` projected by a single linear
      ``f_2`` (Eq. 4); no concat-u/x bypass and no separate FFN.
    - External residual ``x + Layer(x)`` is applied around each layer in
      :class:`HSTUBlock`.
    - Output projection is **tied** with the token embedding by default.

    Parameters
    ----------
    vocab_size : int
        Vocabulary size (items incl. PAD=0).
    d_model : int, default=512
        Hidden dimension.
    n_heads : int, default=8
        Attention heads.
    n_layers : int, default=4
        Number of stacked HSTU layers.
    dqk : int, default=64
        Query/key dim per head.
    dv : int, default=64
        Value/u dim per head.
    max_seq_len : int, default=256
        Maximum sequence length. ``forward`` raises ``ValueError`` if input
        ``seq_len`` exceeds this.
    dropout : float, default=0.1
        Dropout rate.
    use_time_embedding : bool, default=True
        Use time-difference embeddings.
    num_time_buckets : int, default=128
        Number of time buckets.
    time_bucket_fn : {'sqrt', 'log'}, default='sqrt'
        Bucketization function for time differences (in minutes). Shared by
        the input-side time embedding and the per-layer ``rab^{p,t}``.
    time_bucket_divisor : float, default=1.0
        Divisor applied after ``sqrt``/``log`` so the bucket index range
        actually utilizes ``[0, num_time_buckets - 1]``. Tune to your dataset's
        time-difference distribution. Shared by the input-side time embedding
        and the per-layer ``rab^{p,t}``.
    time_bucket_unit : {'minutes', 'seconds'}, default='minutes'
        Unit used before time bucketization. ``'seconds'`` matches Meta's
        public HSTU MovieLens configs; ``'minutes'`` preserves the earlier
        Torch-Rechub behavior.
    tie_embeddings : bool, default=True
        Tie the output projection weight to the token embedding weight.
    score_norm : {'none', 'l2'}, default='none'
        Optional normalization before item scoring. ``'l2'`` matches the
        public HSTU configs' normalized dot-product retrieval setting.
    temperature : float, default=1.0
        Logit temperature applied after item scoring.
    use_output_bias : bool, default=True
        Whether to include an output bias in item logits.
    scale_input_embedding : bool, default=False
        Whether to multiply token embeddings by ``sqrt(d_model)`` before
        adding positional/time embeddings.

    Notes
    -----
    ``time_diffs`` semantics: per-position seconds delta from a single anchor
    (e.g. ``query_time - timestamps[i]``), following the Meta reference.
    Adjacent-step ``t[i] - t[i-1]`` deltas also work, but the bucket
    distribution will be different; set ``time_bucket_divisor`` and
    ``time_bucket_unit`` accordingly.
    ``time_diffs=None`` falls back to all-zero deltas (no temporal signal).

    Shape
    -----
    Input
        x : ``(batch_size, seq_len)``
        time_diffs : ``(batch_size, seq_len)``, optional (seconds).
    Output
        logits : ``(batch_size, seq_len, vocab_size)``
    """

    def __init__(
        self,
        vocab_size,
        d_model=512,
        n_heads=8,
        n_layers=4,
        dqk=64,
        dv=64,
        max_seq_len=256,
        dropout=0.1,
        use_time_embedding=True,
        num_time_buckets=128,
        time_bucket_fn='sqrt',
        time_bucket_divisor=1.0,
        time_bucket_unit='minutes',
        tie_embeddings=True,
        score_norm='none',
        temperature=1.0,
        use_output_bias=True,
        scale_input_embedding=False,
        l2_norm_eps=1e-6,
    ):
        super().__init__()
        if score_norm not in ('none', 'l2'):
            raise ValueError("score_norm must be 'none' or 'l2'")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        self.use_time_embedding = use_time_embedding
        self.num_time_buckets = num_time_buckets
        self.time_bucket_fn = time_bucket_fn
        self.time_bucket_divisor = time_bucket_divisor
        self.time_bucket_unit = time_bucket_unit
        self.tie_embeddings = tie_embeddings
        self.score_norm = score_norm
        self.temperature = temperature
        self.use_output_bias = use_output_bias
        self.scale_input_embedding = scale_input_embedding
        self.l2_norm_eps = l2_norm_eps

        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)

        if use_time_embedding:
            # Bucket 0 is the smallest legal time-diff bucket, not PAD.
            self.time_embedding = nn.Embedding(num_time_buckets, d_model)

        self.hstu_block = HSTUBlock(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dqk=dqk,
            dv=dv,
            dropout=dropout,
            max_seq_len=max_seq_len,
            num_time_buckets=num_time_buckets,
            time_bucket_fn=time_bucket_fn,
            time_bucket_divisor=time_bucket_divisor,
            time_bucket_unit=time_bucket_unit,
        )

        if tie_embeddings:
            # Reuse ``token_embedding.weight``; only the bias is a separate param.
            self.output_bias = nn.Parameter(torch.zeros(vocab_size)) if use_output_bias else None
            self.output_projection = None
        else:
            self.output_projection = nn.Linear(d_model, vocab_size, bias=use_output_bias)
            self.output_bias = None

        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        """Initialize model parameters.

        Xavier-uniform for 2D weights, zeros for biases. After init, force the
        ``padding_idx=0`` row of ``token_embedding`` back to zero (the xavier
        pass overwrites the zero row that ``nn.Embedding(..., padding_idx=0)``
        sets at construction time).
        """
        for name, param in self.named_parameters():
            if 'weight' in name and len(param.shape) > 1:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
        with torch.no_grad():
            self.token_embedding.weight[0].zero_()

    def _time_diff_to_bucket(self, time_diffs):
        """Map raw time differences (seconds) to bucket indices.

        Time deltas are optionally converted to minutes, passed through
        ``sqrt`` or ``log``, then divided by ``time_bucket_divisor`` and
        clipped to ``[0, num_time_buckets-1]``.
        """
        time_diffs = time_diffs.float()
        if self.time_bucket_unit == 'minutes':
            time_diffs = time_diffs / 60.0
        time_diffs = torch.clamp(time_diffs, min=1e-6)

        if self.time_bucket_fn == 'sqrt':
            buckets = torch.sqrt(time_diffs)
        elif self.time_bucket_fn == 'log':
            buckets = torch.log(time_diffs)
        else:
            raise ValueError(f"Unsupported time_bucket_fn: {self.time_bucket_fn}")

        buckets = (buckets / self.time_bucket_divisor).clamp(min=0, max=self.num_time_buckets - 1)
        return buckets.long()

    def forward(self, x, time_diffs=None):
        """Forward pass.

        Args:
            x (Tensor): Input token ids, shape ``(batch_size, seq_len)``. ``0``
                is treated as PAD.
            time_diffs (Tensor, optional): Time differences in seconds, shape
                ``(batch_size, seq_len)``. If ``None`` and
                ``use_time_embedding=True``, all-zero deltas are used (no
                temporal signal).

        Returns:
            Tensor: Logits over the vocabulary, shape
                ``(batch_size, seq_len, vocab_size)``.
        """
        batch_size, seq_len = x.shape
        if seq_len > self.max_seq_len:
            raise ValueError(f"Input seq_len ({seq_len}) exceeds max_seq_len ({self.max_seq_len}). "
                             f"Either truncate the input or rebuild the model with a larger max_seq_len.")

        padding_mask = x.ne(0)  # (B, L) — True for valid tokens

        token_emb = self.token_embedding(x)  # (B, L, D)
        if self.scale_input_embedding:
            token_emb = token_emb * (self.d_model**0.5)
        positions = torch.arange(seq_len, dtype=torch.long, device=x.device)
        pos_emb = self.position_embedding(positions)  # (L, D)

        embeddings = token_emb + pos_emb.unsqueeze(0)

        if self.use_time_embedding:
            if time_diffs is None:
                time_diffs = torch.zeros(batch_size, seq_len, dtype=torch.long, device=x.device)
            time_buckets = self._time_diff_to_bucket(time_diffs)
            embeddings = embeddings + self.time_embedding(time_buckets)

        # Zero out padded positions so position/time embeddings cannot leak
        # signal through PAD rows.
        embeddings = embeddings * padding_mask.unsqueeze(-1).to(embeddings.dtype)

        embeddings = self.dropout(embeddings)

        hstu_output = self.hstu_block(embeddings, padding_mask=padding_mask, time_diffs=time_diffs)
        hstu_output = hstu_output * padding_mask.unsqueeze(-1).to(hstu_output.dtype)

        if self.tie_embeddings:
            output_weight = self.token_embedding.weight
            output_bias = self.output_bias
        else:
            output_weight = self.output_projection.weight
            output_bias = self.output_projection.bias

        if self.score_norm == 'l2':
            hstu_output = F.normalize(hstu_output, p=2, dim=-1, eps=self.l2_norm_eps)
            output_weight = F.normalize(output_weight, p=2, dim=-1, eps=self.l2_norm_eps)

        logits = F.linear(hstu_output, output_weight, output_bias)
        if self.temperature != 1.0:
            logits = logits / self.temperature

        return logits
