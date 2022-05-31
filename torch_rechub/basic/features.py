from ..utils.data import get_auto_embedding_dim


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
    """

    def __init__(self, name, vocab_size, embed_dim=None, pooling="mean", shared_with=None):
        self.name = name
        self.vocab_size = vocab_size
        if embed_dim == None:
            self.embed_dim = get_auto_embedding_dim(vocab_size)
        else:
            self.embed_dim = embed_dim
        self.pooling = pooling
        self.shared_with = shared_with


class SparseFeature(object):
    """The Feature Class for Sparse feature.

    Args:
        name (str): feature's name.
        vocab_size (int): vocabulary size of embedding table.
        embed_dim (int): embedding vector's length
        shared_with (str): the another feature name which this feature will shared with embedding.
    """

    def __init__(self, name, vocab_size, embed_dim=None, shared_with=None):
        self.name = name
        self.vocab_size = vocab_size
        if embed_dim == None:
            self.embed_dim = get_auto_embedding_dim(vocab_size)
        else:
            self.embed_dim = embed_dim
        self.shared_with = shared_with


class DenseFeature(object):
    """The Feature Class for Dense feature.

    Args:
        name (str): feature's name.
        embed_dim (int): embedding vector's length, the value fixed `1`.
    """

    def __init__(self, name):
        self.name = name
        self.embed_dim = 1