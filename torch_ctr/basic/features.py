from .utils import get_auto_embedding_dim


class SequenceFeature():
    """
    """

    def __init__(self, name, vocab_size, embed_dim, pooling="mean", shared_with=None):
        """
        :param name: String, means sparse feature's name
        :param vocab_size: Integer, means vocabulary size
        :param embed_dim: Integer, means embedding vector's length
        """
        self.name = name
        self.vocab_size = vocab_size
        if embed_dim == None:
            self.embed_dim = get_auto_embedding_dim(vocab_size)
        else:
            self.embed_dim = embed_dim
        self.pooling = pooling
        self.shared_with = shared_with


class SparseFeature():
    """
    """

    def __init__(self, name, vocab_size, embed_dim=None):
        """
        :param name: String, means sparse feature's name
        :param vocab_size: Integer, means vocabulary size
        :param embed_dim: Integer, means embedding vector's length
        """
        self.name = name
        self.vocab_size = vocab_size
        if embed_dim == None:
            self.embed_dim = get_auto_embedding_dim(vocab_size)
        else:
            self.embed_dim = embed_dim


class DenseFeature():
    """
    """

    def __init__(self, name):
        """
        :param name: String, means dense feature's name
        """
        self.name = name
        self.embed_dim = 1