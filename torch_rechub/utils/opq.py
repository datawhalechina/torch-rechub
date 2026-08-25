"""OPQ-based semantic-ID tokenizer.

Turns dense item embeddings into short tuples of discrete codes with Optimized
Product Quantization (OPQ). Unlike the residual quantization used by RQ-VAE /
TIGER, the codes produced here are **unordered**: digit ``j`` quantizes its own
subspace of the (rotated) embedding, so no digit is a refinement of the
previous one. This is what lets RPG predict all digits in parallel.

Reference
---------
Hou et al., "Generating Long Semantic IDs in Parallel for Recommendation",
KDD 2025. https://arxiv.org/abs/2506.05781
Official implementation: https://github.com/facebookresearch/RPG_KDD2025
"""

import json
import math

import numpy as np
import torch
from sklearn.decomposition import PCA


def _load_faiss():
    try:
        import faiss
    except ModuleNotFoundError as exc:
        raise ImportError("OPQTokenizer requires faiss. Install with `pip install torch-rechub[generative]`.") from exc
    return faiss


class OPQTokenizer:
    """Quantize item embeddings into unordered semantic IDs with FAISS OPQ.

    The token vocabulary is laid out so that every digit owns a disjoint slice
    of ids, which lets a model keep one embedding table for all digits::

        0                              -> PAD
        1 .. codebook_size             -> digit 0
        codebook_size+1 .. 2*codebook_size -> digit 1
        ...
        n_codebook*codebook_size + 1   -> EOS

    Parameters
    ----------
    n_codebook : int, default=32
        Number of digits ``m`` per semantic ID.
    codebook_size : int, default=256
        Number of codes ``M`` per digit. Must be a power of two, since FAISS
        product quantizers are parameterized by bits per sub-quantizer.
    pca_dim : int, default=128
        Whitened-PCA dimension applied to the embeddings before quantization.
        Set to ``0`` to skip PCA.
    use_gpu : bool, default=False
        Train the FAISS index on GPU. Requires ``faiss-gpu``.
    gpu_id : int, default=0
        GPU used when ``use_gpu`` is set.
    n_threads : int, default=32
        FAISS OpenMP thread count.

    Attributes
    ----------
    codes : numpy.ndarray or None
        ``(n_items - 1, n_codebook)`` raw codes in ``[0, codebook_size)``, one
        row per item id ``1..n_items-1``. ``None`` before :meth:`fit`.

    Examples
    --------
    >>> import numpy as np
    >>> embeddings = np.random.randn(500, 64).astype(np.float32)
    >>> tokenizer = OPQTokenizer(n_codebook=4, codebook_size=16, pca_dim=32)
    >>> tokenizer.fit(embeddings)  # doctest: +SKIP
    >>> tokenizer.item_tokens().shape  # doctest: +SKIP
    torch.Size([501, 4])
    """

    def __init__(self, n_codebook=32, codebook_size=256, pca_dim=128, use_gpu=False, gpu_id=0, n_threads=32):
        bits = math.log2(codebook_size)
        if not bits.is_integer():
            raise ValueError(f"codebook_size must be a power of two, got {codebook_size}")
        self.n_codebook = n_codebook
        self.codebook_size = codebook_size
        self.n_bits = int(bits)
        self.pca_dim = pca_dim
        self.use_gpu = use_gpu
        self.gpu_id = gpu_id
        self.n_threads = n_threads
        self.index_factory = f"OPQ{n_codebook},IVF1,PQ{n_codebook}x{self.n_bits}"
        self.codes = None

    @property
    def n_items(self):
        """Number of item ids including PAD at index 0."""
        return len(self.codes) + 1

    @property
    def eos_token(self):
        return self.n_codebook * self.codebook_size + 1

    @property
    def vocab_size(self):
        return self.eos_token + 1

    def fit(self, embeddings, train_mask=None):
        """Learn the quantizer and encode every item.

        Parameters
        ----------
        embeddings : numpy.ndarray
            ``(n_items - 1, dim)`` item embeddings, row ``i`` belonging to item
            id ``i + 1`` (id ``0`` is PAD and has no embedding).
        train_mask : numpy.ndarray, optional
            Boolean mask over rows selecting the items used to *train* the
            quantizer. All items are encoded regardless. Restricting training
            to items seen during training avoids leaking test-only items into
            the codebooks.

        Returns
        -------
        OPQTokenizer
            ``self``, so calls can be chained.

        Notes
        -----
        FAISS learns the OPQ rotation with a 256-centroid quantizer whatever
        ``codebook_size`` is, so at least 256 training rows are required.
        """
        faiss = _load_faiss()

        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        if self.pca_dim > 0:
            embeddings = PCA(n_components=self.pca_dim, whiten=True).fit_transform(embeddings)
            embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)

        faiss.omp_set_num_threads(self.n_threads)
        index = faiss.index_factory(embeddings.shape[1], self.index_factory, faiss.METRIC_INNER_PRODUCT)
        if self.use_gpu:
            resources = faiss.StandardGpuResources()
            options = faiss.GpuClonerOptions()
            options.useFloat16 = self.n_codebook >= 56
            index = faiss.index_cpu_to_gpu(resources, self.gpu_id, index, options)

        train_embeddings = embeddings if train_mask is None else embeddings[train_mask]
        index.train(train_embeddings)
        index.add(embeddings)
        if self.use_gpu:
            index = faiss.index_gpu_to_cpu(index)

        self.codes = self._read_pq_codes(faiss, index)
        return self

    def _read_pq_codes(self, faiss, index):
        """Extract packed PQ codes from the single inverted list.

        The factory string pins ``nlist=1``, so every vector lands in list 0 in
        insertion order and the codes line up with the input rows. FAISS packs
        the ``n_bits``-wide codes into bytes, hence the bitstring reader.
        """
        invlists = faiss.extract_index_ivf(faiss.downcast_index(index.index)).invlists
        list_size = invlists.list_size(0)
        packed = faiss.rev_swig_ptr(invlists.get_codes(0), list_size * invlists.code_size)
        packed = packed.reshape(list_size, invlists.code_size)

        codes = np.empty((list_size, self.n_codebook), dtype=np.int64)
        for i, row in enumerate(packed):
            reader = faiss.BitstringReader(faiss.swig_ptr(row), invlists.code_size)
            for digit in range(self.n_codebook):
                codes[i, digit] = reader.read(self.n_bits)
        return codes

    def item_tokens(self):
        """Return the token-id table consumed by the model.

        Returns
        -------
        torch.LongTensor
            ``(n_items, n_codebook)`` token ids, with row ``0`` reserved for
            PAD and filled with zeros.
        """
        if self.codes is None:
            raise RuntimeError("Call fit() or load() before item_tokens().")
        offsets = np.arange(self.n_codebook) * self.codebook_size + 1
        tokens = torch.from_numpy(self.codes + offsets)
        return torch.cat([torch.zeros(1, self.n_codebook, dtype=torch.long), tokens])

    def collision_rate(self):
        """Fraction of items whose semantic ID is shared with another item."""
        unique = np.unique(self.codes, axis=0)
        return 1.0 - len(unique) / len(self.codes)

    def save(self, path):
        """Write the codes and their layout to a JSON file."""
        with open(path, "w") as f:
            json.dump({"n_codebook": self.n_codebook, "codebook_size": self.codebook_size, "pca_dim": self.pca_dim, "codes": self.codes.tolist()}, f)

    @classmethod
    def load(cls, path):
        """Rebuild a fitted tokenizer from :meth:`save` output."""
        with open(path, "r") as f:
            state = json.load(f)
        tokenizer = cls(n_codebook=state["n_codebook"], codebook_size=state["codebook_size"], pca_dim=state["pca_dim"])
        tokenizer.codes = np.asarray(state["codes"], dtype=np.int64)
        return tokenizer
