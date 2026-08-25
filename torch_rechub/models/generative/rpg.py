"""RPG: Recommendation with Parallel Generation of long semantic IDs.

Reference
---------
Hou et al., "Generating Long Semantic IDs in Parallel for Recommendation",
KDD 2025. https://arxiv.org/abs/2506.05781
Official implementation: https://github.com/facebookresearch/RPG_KDD2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Config, GPT2Model


class ResBlock(nn.Module):
    """Residual block ``x + SiLU(Linear(x))``, an identity map at init.

    Both the weight and the bias start at zero, so the block leaves the
    backbone output untouched until training moves it away from identity.

    Parameters
    ----------
    hidden_size : int
        Input and output dimension.
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        self.act = nn.SiLU()

    def forward(self, x):
        return x + self.act(self.linear(x))


class RPGModel(nn.Module):
    """Predict all digits of an item's semantic ID in parallel.

    Each item is quantized by :class:`~torch_rechub.utils.opq.OPQTokenizer`
    into ``n_digit`` **unordered** codes. Because no digit refines another,
    the whole semantic ID can be predicted in a single forward pass instead of
    being decoded autoregressively as in TIGER, which is what allows semantic
    IDs to grow from 4 to 64 digits.

    - An item is represented by mean-pooling the embeddings of its
      ``n_digit`` tokens, so it occupies one position in the sequence.
    - A causal GPT-2 backbone encodes the user history.
    - ``n_digit`` independent :class:`ResBlock` heads project the hidden state
      into one query per digit. Each query is scored against its own codebook
      by temperature-scaled cosine similarity, and the heads are trained with
      a cross-entropy per digit (multi-token prediction).
    - At inference an item's score is the mean log-probability of its own
      tokens. Scoring every item is exact but linear in the catalog size, so
      :meth:`generate` can instead walk an item-item similarity graph built
      from the learned codebooks, visiting only a fraction of the items.

    Parameters
    ----------
    item_tokens : torch.LongTensor
        ``(n_items, n_digit)`` token-id table from
        :meth:`~torch_rechub.utils.opq.OPQTokenizer.item_tokens`. Row ``0`` is
        PAD.
    codebook_size : int, default=256
        Number of codes per digit.
    n_embd : int, default=448
        Hidden dimension.
    n_layer : int, default=2
        Number of transformer layers.
    n_head : int, default=4
        Number of attention heads.
    n_inner : int, default=1024
        Feed-forward dimension.
    max_seq_len : int, default=50
        Maximum number of items per sequence.
    resid_pdrop, embd_pdrop, attn_pdrop : float
        GPT-2 dropout rates. The defaults follow the paper, which relies on
        heavy embedding/attention dropout to regularize a small backbone.
    temperature : float, default=0.07
        Temperature dividing the cosine logits.
    layer_norm_epsilon : float, default=1e-12
        GPT-2 layer-norm epsilon.
    initializer_range : float, default=0.02
        GPT-2 weight init range.

    Shape
    -----
    Input
        input_ids : ``(batch_size, seq_len)`` item ids, right-padded with ``0``
        attention_mask : ``(batch_size, seq_len)``
        labels : ``(batch_size, seq_len)`` next-item ids, ``-100`` to ignore
    Output
        states : ``(batch_size, seq_len, n_digit, n_embd)``

    Examples
    --------
    >>> import torch
    >>> item_tokens = torch.randint(1, 17, (100, 4))
    >>> item_tokens[0] = 0
    >>> model = RPGModel(item_tokens, codebook_size=16, n_embd=32, n_inner=64)
    >>> input_ids = torch.randint(1, 100, (2, 5))
    >>> mask = torch.ones(2, 5, dtype=torch.long)
    >>> states, loss = model(input_ids, mask, labels=input_ids)
    >>> states.shape
    torch.Size([2, 5, 4, 32])
    """

    def __init__(
        self,
        item_tokens,
        codebook_size=256,
        n_embd=448,
        n_layer=2,
        n_head=4,
        n_inner=1024,
        max_seq_len=50,
        resid_pdrop=0.0,
        embd_pdrop=0.5,
        attn_pdrop=0.5,
        temperature=0.07,
        layer_norm_epsilon=1e-12,
        initializer_range=0.02,
    ):
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        n_items, n_digit = item_tokens.shape

        self.n_items = n_items
        self.n_digit = n_digit
        self.codebook_size = codebook_size
        self.n_embd = n_embd
        self.temperature = temperature

        # PAD occupies id 0 and EOS the last id, so the digit codebooks are the
        # contiguous block ``[1, n_digit * codebook_size]``.
        self.eos_token = n_digit * codebook_size + 1
        self.register_buffer('item_tokens', item_tokens.long(), persistent=True)
        self.register_buffer('digit_offsets', torch.arange(n_digit) * codebook_size + 1, persistent=False)

        self.gpt2 = GPT2Model(
            GPT2Config(
                vocab_size=self.eos_token + 1,
                n_positions=max_seq_len,
                n_embd=n_embd,
                n_layer=n_layer,
                n_head=n_head,
                n_inner=n_inner,
                activation_function='gelu_new',
                resid_pdrop=resid_pdrop,
                embd_pdrop=embd_pdrop,
                attn_pdrop=attn_pdrop,
                layer_norm_epsilon=layer_norm_epsilon,
                initializer_range=initializer_range,
                eos_token_id=self.eos_token,
            )
        )
        self.pred_heads = nn.ModuleList([ResBlock(n_embd) for _ in range(n_digit)])
        self.adjacency = None

    # =========================================================
    # Encoding and loss
    # =========================================================
    def _digit_embeddings(self):
        """Codebook embeddings reshaped to ``(n_digit, codebook_size, n_embd)``."""
        return self.gpt2.wte.weight[1:self.eos_token].view(self.n_digit, self.codebook_size, self.n_embd)

    def _digit_logits(self, states):
        """Score per-digit queries against their own codebook.

        Both sides are L2-normalized, so the logits are cosine similarities
        scaled by ``1 / temperature``.

        Parameters
        ----------
        states : torch.Tensor
            ``(..., n_digit, n_embd)`` per-digit queries.

        Returns
        -------
        torch.Tensor
            ``(..., n_digit, codebook_size)`` logits.
        """
        states = F.normalize(states, dim=-1)
        digit_emb = F.normalize(self._digit_embeddings(), dim=-1)
        return torch.einsum('...md,mkd->...mk', states, digit_emb) / self.temperature

    def forward(self, input_ids, attention_mask, labels=None):
        """Encode a history and optionally compute the multi-token loss.

        Parameters
        ----------
        input_ids : torch.LongTensor
            ``(batch_size, seq_len)`` item ids, right-padded with ``0``.
        attention_mask : torch.LongTensor
            ``(batch_size, seq_len)``, ``1`` on real items.
        labels : torch.LongTensor, optional
            ``(batch_size, seq_len)`` next-item ids, ``-100`` where no loss
            should be computed.

        Returns
        -------
        states : torch.Tensor
            ``(batch_size, seq_len, n_digit, n_embd)`` per-digit queries.
        loss : torch.Tensor or None
            Mean cross-entropy over digits, ``None`` when ``labels`` is
            ``None``.
        """
        item_embs = self.gpt2.wte(self.item_tokens[input_ids]).mean(dim=-2)
        hidden = self.gpt2(inputs_embeds=item_embs, attention_mask=attention_mask).last_hidden_state
        states = torch.stack([head(hidden) for head in self.pred_heads], dim=-2)
        if labels is None:
            return states, None

        labels = labels.reshape(-1)
        keep = labels != -100
        logits = self._digit_logits(states.reshape(-1, self.n_digit, self.n_embd)[keep])
        targets = self.item_tokens[labels[keep]] - self.digit_offsets
        loss = F.cross_entropy(logits.reshape(-1, self.codebook_size), targets.reshape(-1))
        return states, loss

    # =========================================================
    # Scoring
    # =========================================================
    def next_token_logits(self, states, seq_lens):
        """Log-probabilities over every codebook token at the last real position.

        Parameters
        ----------
        states : torch.Tensor
            ``(batch_size, seq_len, n_digit, n_embd)`` from :meth:`forward`.
        seq_lens : torch.LongTensor
            ``(batch_size,)`` number of real items per row.

        Returns
        -------
        torch.Tensor
            ``(batch_size, n_digit * codebook_size)`` log-probabilities, with
            each digit's block normalized independently.
        """
        index = (seq_lens - 1).view(-1, 1, 1, 1).expand(-1, 1, self.n_digit, self.n_embd)
        last = states.gather(1, index).squeeze(1)
        return self._digit_logits(last).log_softmax(dim=-1).flatten(1)

    def score_all_items(self, token_logits):
        """Score the whole catalog: the mean log-probability of each item's tokens.

        Parameters
        ----------
        token_logits : torch.Tensor
            ``(batch_size, n_digit * codebook_size)`` from
            :meth:`next_token_logits`.

        Returns
        -------
        torch.Tensor
            ``(batch_size, n_items - 1)`` scores for item ids ``1..n_items-1``.
        """
        tokens = self.item_tokens[1:] - 1
        scores = token_logits[:, tokens.reshape(-1)]
        return scores.view(len(token_logits), -1, self.n_digit).mean(-1)

    def score_candidates(self, token_logits, item_ids):
        """Score a per-row candidate set.

        Parameters
        ----------
        token_logits : torch.Tensor
            ``(batch_size, n_digit * codebook_size)``.
        item_ids : torch.LongTensor
            ``(batch_size, n_candidates)`` item ids.

        Returns
        -------
        torch.Tensor
            ``(batch_size, n_candidates)`` scores.
        """
        tokens = self.item_tokens[item_ids] - 1
        scores = token_logits.gather(1, tokens.flatten(1))
        return scores.view(tokens.shape).mean(-1)

    # =========================================================
    # Graph-constrained decoding
    # =========================================================
    def build_decoding_graph(self, n_edges=50, chunk_size=1024):
        """Build the item-item similarity graph used by :meth:`generate`.

        Item similarity is the sum over digits of the cosine similarity
        between the two items' codebook embeddings for that digit, mapped to
        ``[0, 1]``. Only the ``n_edges`` nearest neighbours of each item are
        kept, and rows are processed in chunks so the dense ``n_items x
        n_items`` matrix is never materialized.

        Parameters
        ----------
        n_edges : int, default=50
            Out-degree of each node.
        chunk_size : int, default=1024
            Number of rows scored at a time.
        """
        digit_emb = F.normalize(self._digit_embeddings(), dim=-1)
        token_sim = 0.5 * (torch.bmm(digit_emb, digit_emb.transpose(1, 2)) + 1.0)
        codes = self.item_tokens[1:] - self.digit_offsets

        neighbors = [torch.zeros(1, n_edges, dtype=torch.long, device=codes.device)]
        for start in range(0, len(codes), chunk_size):
            block = codes[start:start + chunk_size]
            sim = torch.zeros(len(block), len(codes), device=digit_emb.device)
            for digit in range(self.n_digit):
                sim += token_sim[digit][block[:, digit]][:, codes[:, digit]]
            # ``+ 1`` shifts a row index back to an item id.
            neighbors.append(sim.topk(n_edges, dim=-1).indices + 1)
        self.adjacency = torch.cat(neighbors)

    @staticmethod
    def _duplicate_mask(values):
        """Mark every occurrence of a value in a row except the first one."""
        ordered, order = values.sort(dim=1)
        repeated = torch.zeros_like(ordered, dtype=torch.bool)
        repeated[:, 1:] = ordered[:, 1:] == ordered[:, :-1]
        return torch.zeros_like(repeated).scatter_(1, order, repeated)

    def graph_propagation(self, token_logits, topk, num_beams=50, propagation_steps=3):
        """Beam search over the decoding graph.

        Starts from a random beam, repeatedly expands it to the neighbours of
        the current nodes, and keeps the ``num_beams`` best-scoring distinct
        items.

        Parameters
        ----------
        token_logits : torch.Tensor
            ``(batch_size, n_digit * codebook_size)``.
        topk : int
            Number of items to return.
        num_beams : int, default=50
            Beam width.
        propagation_steps : int, default=3
            Number of expansion rounds.

        Returns
        -------
        torch.LongTensor
            ``(batch_size, topk)`` item ids ordered by decreasing score.

        Notes
        -----
        Sets ``self.n_visited_items`` to the mean number of distinct items
        scored per query, the efficiency figure the paper reports against
        exhaustive ranking.
        """
        if self.adjacency is None:
            raise RuntimeError("Call build_decoding_graph() before graph_propagation().")
        if topk > num_beams:
            raise ValueError(f"topk ({topk}) cannot exceed num_beams ({num_beams})")

        batch_size = len(token_logits)
        beams = torch.randint(1, self.n_items, (batch_size, num_beams), device=token_logits.device)
        visited = torch.zeros(batch_size, self.n_items, dtype=torch.bool, device=token_logits.device)
        visited.scatter_(1, beams, True)
        for _ in range(propagation_steps):
            candidates = self.adjacency[beams].flatten(1)
            visited.scatter_(1, candidates, True)
            scores = self.score_candidates(token_logits, candidates)
            # A neighbour reachable from several beams must not fill the beam twice.
            scores = scores.masked_fill(self._duplicate_mask(candidates), -torch.inf)
            beams = candidates.gather(1, scores.topk(num_beams, dim=-1).indices)
        self.n_visited_items = visited.sum(1).float().mean().item()
        return beams[:, :topk]

    @torch.no_grad()
    def generate(self, input_ids, attention_mask, seq_lens, topk=10, use_graph=False, num_beams=50, propagation_steps=3):
        """Recommend the ``topk`` next items for each history.

        Parameters
        ----------
        input_ids : torch.LongTensor
            ``(batch_size, seq_len)`` item ids, right-padded with ``0``.
        attention_mask : torch.LongTensor
            ``(batch_size, seq_len)``.
        seq_lens : torch.LongTensor
            ``(batch_size,)`` number of real items per row.
        topk : int, default=10
            Number of items to return.
        use_graph : bool, default=False
            Walk the decoding graph instead of scoring the whole catalog.
            Requires :meth:`build_decoding_graph` to have been called.
        num_beams, propagation_steps : int
            Forwarded to :meth:`graph_propagation`.

        Returns
        -------
        torch.LongTensor
            ``(batch_size, topk)`` item ids ordered by decreasing score.
        """
        states, _ = self.forward(input_ids, attention_mask)
        token_logits = self.next_token_logits(states, seq_lens)
        if use_graph:
            return self.graph_propagation(token_logits, topk, num_beams, propagation_steps)
        # ``+ 1`` shifts a column index back to an item id.
        return self.score_all_items(token_logits).topk(topk, dim=-1).indices + 1
