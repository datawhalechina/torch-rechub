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

    def forward(self, x, rel_pos_bias=None, padding_mask=None):
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

        allowed = causal_mask.unsqueeze(0).unsqueeze(0)
        if padding_mask is not None:
            allowed = allowed & padding_mask[:, None, None, :]
        # An all-PAD query has no legal keys: zero its weights after softmax.
        scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = attn_weights.masked_fill(~allowed, 0)
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
        if padding_mask is not None:
            x = x * padding_mask.unsqueeze(-1).to(x.dtype)

        return x


class HLLMModel(nn.Module):
    """HLLM: Hierarchical Large Language Model for Recommendation.

    This is a lightweight implementation of HLLM that uses pre-computed item
    embeddings as input. The original ByteDance HLLM uses end-to-end training
    with both Item LLM and User LLM, but this implementation focuses on the
    sequence encoder for resource efficiency. By default it is an HLLM-inspired
    baseline, not a pretrained User LLM or full end-to-end paper reproduction.

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
            Row ``i`` must correspond to vocab token ``i`` and row 0 is PAD.
        vocab_size (int): Vocabulary size including PAD.
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
        temperature (float): Temperature for cos-sim logits. Default: 0.07
            (matches official ByteDance HLLM logit_scale = log(1/0.07)).
            item_embeddings and user representation are both L2-normalized,
            so logits = cos(x, emb) / temperature, bounded in [-1/T, +1/T].
            Loss functions should be configured with their own temperature=1.0
            to avoid double scaling.
        user_model (nn.Module, optional): Supplied pretrained causal base model,
            e.g. AutoModel, returning last_hidden_state from inputs_embeds.
            Its hidden size must match d_model. A causal-LM vocabulary head is
            not needed. No checkpoint is downloaded by this class.
        project_items (bool): Default False preserves the dimension check.
            True allows a small trainable projection from cached LLM embeddings
            to d_model; this is a resource-saving approximation, not paper HLLM.
        learn_temperature (bool): Default False preserves fixed temperature.
            True learns a logit scale clamped to [1, 100], like the reference.
    """

    def __init__(
        self,
        item_embeddings,
        vocab_size,
        d_model=512,
        n_heads=8,
        n_layers=4,
        max_seq_len=256,
        dropout=0.1,
        use_rel_pos_bias=True,
        use_time_embedding=True,
        num_time_buckets=2048,
        time_bucket_fn='sqrt',
        temperature=0.07,
        user_model=None,
        project_items=False,
        learn_temperature=False
    ):
        super().__init__()
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
        self.temperature = temperature
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / temperature))) if learn_temperature else None

        # Load item embeddings
        if isinstance(item_embeddings, str):
            item_embeddings = torch.load(item_embeddings)

        if item_embeddings.shape[0] != vocab_size:
            raise ValueError(f"item_embeddings.shape[0]={item_embeddings.shape[0]} "
                             f"!= vocab_size={vocab_size}. "
                             "Embedding tensor must be indexed by token_id "
                             "(row i = embedding of vocab token i, row 0 = PAD).")
        if item_embeddings.shape[1] != d_model and not project_items:
            raise ValueError(f"item_embeddings.shape[1]={item_embeddings.shape[1]} != d_model={d_model}; set project_items=True to learn a projection")
        self.item_projection = nn.Identity() if item_embeddings.shape[1] == d_model else nn.Linear(item_embeddings.shape[1], d_model, bias=False)
        # Optional pretrained causal base model accepting inputs_embeds.
        # Do not load checkpoints or execute remote model code implicitly.
        self.user_model = user_model
        if user_model is not None and user_model.config.hidden_size != d_model:
            raise ValueError("user_model.config.hidden_size must match d_model")

        # Preserve raw input magnitudes for an external pretrained User LLM.
        # The historical small-encoder path keeps its normalized cache.
        item_embeddings = item_embeddings.float()
        if user_model is None:
            item_embeddings = F.normalize(item_embeddings, dim=-1, eps=1e-8)

        # Register as buffer (not trainable)
        self.register_buffer('item_embeddings', item_embeddings)

        # Positional embedding
        self.position_embedding = nn.Embedding(max_seq_len, d_model) if user_model is None else None

        # Time embedding
        if use_time_embedding and user_model is None:
            self.time_embedding = nn.Embedding(num_time_buckets + 1, d_model, padding_idx=0)

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([HLLMTransformerBlock(d_model=d_model, n_heads=n_heads, dropout=dropout) for _ in range(n_layers)] if user_model is None else [])

        # Relative position bias
        self.use_rel_pos_bias = use_rel_pos_bias
        if use_rel_pos_bias and user_model is None:
            self.rel_pos_bias = RelPosBias(n_heads, max_seq_len)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        """Initialize model parameters."""
        for name, param in self.named_parameters():
            if name.startswith('user_model.'):
                continue  # Keep supplied pretrained weights (including LoRA).
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

    def encode(self, seq_tokens, time_diffs=None):
        """Encode history into states, masking PAD keys and PAD outputs.
        
        Args:
            seq_tokens (Tensor): Item token IDs of shape (B, L).
            time_diffs (Tensor, optional): Time differences in seconds of shape (B, L).
            
        Returns:
            Tensor: States of shape (B, L, d_model).
        """
        batch_size, seq_len = seq_tokens.shape
        if seq_len > self.max_seq_len:
            raise ValueError("Sequence exceeds max_seq_len")
        padding_mask = seq_tokens.ne(0)

        # Look up item embeddings
        item_emb = self.item_projection(self.item_embeddings[seq_tokens])
        if self.user_model is not None:
            # Position/time modelling belongs to the supplied causal LLM.
            # Cached embeddings/projection may be FP32 while a pretrained
            # User LLM uses FP16/BF16; align inputs without breaking gradients.
            item_emb = item_emb.to(dtype=self.user_model.get_input_embeddings().weight.dtype)
            outputs = self.user_model(inputs_embeds=item_emb, attention_mask=padding_mask.long(), use_cache=False, return_dict=True)
            return outputs.last_hidden_state * padding_mask.unsqueeze(-1).to(item_emb.dtype)

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
        embeddings = embeddings * padding_mask.unsqueeze(-1).to(embeddings.dtype)

        # Get relative position bias
        rel_pos_bias = None
        if self.use_rel_pos_bias:
            rel_pos_bias = self.rel_pos_bias(seq_len)

        # Pass through transformer blocks
        x = embeddings
        for block in self.transformer_blocks:
            x = block(x, rel_pos_bias=rel_pos_bias, padding_mask=padding_mask)
        return x

    def score_items(self, states, item_ids):
        """Cosine scores on a candidate subset; item table remains frozen."""
        # Scoring head: cosine similarity with normalized item embeddings
        # x: (B, L, D), item_embeddings: (V, D)
        states = F.normalize(states.float(), dim=-1, eps=1e-8)
        items = self.get_item_embeddings(item_ids)
        if self.logit_scale is not None:
            with torch.no_grad():
                self.logit_scale.clamp_(0, math.log(100))
            scale = self.logit_scale.exp()
        else:
            scale = 1.0 / self.temperature
        logits = torch.matmul(states, items.t()) * scale

        return logits

    def get_item_embeddings(self, item_ids):
        """Normalized candidate vectors, including the optional projection."""
        return F.normalize(self.item_projection(self.item_embeddings[item_ids]).float(), dim=-1, eps=1e-8)

    def forward(self, seq_tokens, time_diffs=None):
        """Preserve the full-vocabulary forward API."""
        return self.score_items(self.encode(seq_tokens, time_diffs), torch.arange(self.vocab_size, device=seq_tokens.device))
