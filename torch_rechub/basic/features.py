from ..utils.data import get_auto_embedding_dim
from .initializers import RandomNormal


class SequenceFeature(object):
    """The Feature Class for Sequence feature or multi-hot feature.
    In recommendation, there are many user behaviour features which we want to take the sequence model
    and tag featurs (multi hot) which we want to pooling. Note that if you use this feature, you must padding
    the feature value before training.

    Args:
        name (str): feature's name.
        vocab_size (int): vocabulary size of embedding table.
        embed_dim (int): embedding vector's length
        pooling (str): pooling method, support `["mean", "sum", "concat"]` (default=`"mean"`)
        shared_with (str): the another feature name which this feature will shared with embedding.
        padding_idx (int, optional): If specified, the entries at padding_idx will be masked 0 in InputMask Layer.
        initializer(Initializer): Initializer the embedding layer weight.
    """

    def __init__(self, name, vocab_size, embed_dim=None, pooling="mean", shared_with=None, padding_idx=None, initializer=RandomNormal(0, 0.0001)):
        self.name = name
        self.vocab_size = vocab_size
        if embed_dim is None:
            self.embed_dim = get_auto_embedding_dim(vocab_size)
        else:
            self.embed_dim = embed_dim
        self.pooling = pooling
        self.shared_with = shared_with
        self.padding_idx = padding_idx
        self.initializer = initializer

    def __repr__(self):
        return f'<SequenceFeature {self.name} with Embedding shape ({self.vocab_size}, {self.embed_dim})>'

    def get_embedding_layer(self):
        if not hasattr(self, 'embed'):
            self.embed = self.initializer(self.vocab_size, self.embed_dim)
        return self.embed


class SparseFeature(object):
    """The Feature Class for Sparse feature.

    Args:
        name (str): feature's name.
        vocab_size (int): vocabulary size of embedding table.
        embed_dim (int): embedding vector's length
        shared_with (str): the another feature name which this feature will shared with embedding.
        padding_idx (int, optional): If specified, the entries at padding_idx will be masked 0 in InputMask Layer.
        initializer(Initializer): Initializer the embedding layer weight.
    """

    def __init__(self, name, vocab_size, embed_dim=None, shared_with=None, padding_idx=None, initializer=RandomNormal(0, 0.0001)):
        self.name = name
        self.vocab_size = vocab_size
        if embed_dim is None:
            self.embed_dim = get_auto_embedding_dim(vocab_size)
        else:
            self.embed_dim = embed_dim
        self.shared_with = shared_with
        self.padding_idx = padding_idx
        self.initializer = initializer

    def __repr__(self):
        return f'<SparseFeature {self.name} with Embedding shape ({self.vocab_size}, {self.embed_dim})>'

    def get_embedding_layer(self):
        if not hasattr(self, 'embed'):
            self.embed = self.initializer(self.vocab_size, self.embed_dim)
        return self.embed


class DenseFeature(object):
    """The Feature Class for Dense feature.

    Args:
        name (str): feature's name.
        embed_dim (int): embedding vector's length, the value fixed `1`. If you put a vector (torch.tensor) , replace the embed_dim with your vector dimension.
    """

    def __init__(self, name, embed_dim=1):
        self.name = name
        self.embed_dim = embed_dim

    def __repr__(self):
        return f'<DenseFeature {self.name}>'
