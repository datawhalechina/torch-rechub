import torch


class RandomNormal(object):
    """Initializer that generates tensors with a normal distribution.

    Args:
        mean (float): the mean of the normal distribution
        std (float): the standard deviation of the normal distribution
    """

    def __init__(self, mean=0.0, std=1.0):
        self.mean = mean
        self.std = std

    def __call__(self, feature=None, weight=None):
        """Returns a embedding layer initialized to random normal values or Fill parameter weights with a normal distribution.

        Args:
            feature (SequenceFeature or SparseFeature, optional): 
                    If feature is not None, return a embedding layer initialized to random normal values. Defaults to None.
            weight (torch.nn.parameter.Parameter, optional): 
                    If weight is not None, fill weight with a normal distribution. Defaults to None.

        Returns:
            torch.nn.Embedding: If feature is not None.
            None: If feature is None.
        """
        if feature != None:
            embed = torch.nn.Embedding(feature.vocab_size, feature.embed_dim)
            weight = embed.weight
        if weight != None:
            torch.nn.init.normal_(weight, self.mean, self.std)
        if feature != None:
            return embed


class RandomUniform(object):
    """Initializer that generates tensors with a uniform distribution.

    Args:
        minval (float): Lower bound of the range of random values of the uniform distribution.
        maxval (float): Upper bound of the range of random values of the uniform distribution.
    """

    def __init__(self, minval=0.0, maxval=1.0):
        self.minval = minval
        self.maxval = maxval

    def __call__(self, feature=None, weight=None):
        if feature != None:
            embed = torch.nn.Embedding(feature.vocab_size, feature.embed_dim)
            weight = embed.weight
        if weight != None:
            torch.nn.init.uniform_(weight, self.minval, self.maxval)
        if feature != None:
            return embed


class XavierNormal(object):
    """Initializer that generates tensors according to the method
    described in `Understanding the difficulty of training deep feedforward
    neural networks` - Glorot, X. & Bengio, Y. (2010), using a uniform
    distribution.

    Args:
        gain (float): stddev = gain*sqrt(2 / (fan_in + fan_out))
    """

    def __init__(self, gain=1.):
        self.gain = gain

    def __call__(self, feature=None, weight=None):
        if feature != None:
            embed = torch.nn.Embedding(feature.vocab_size, feature.embed_dim)
            weight = embed.weight
        if weight != None:
            torch.nn.init.xavier_normal_(weight, self.gain)
        if feature != None:
            return embed


class XavierUniform(object):
    """Initializer that generates tensors according to the method
    described in `Understanding the difficulty of training deep feedforward
    neural networks` - Glorot, X. & Bengio, Y. (2010), using a uniform
    distribution.

    Args:
        gain (float): stddev = gain*sqrt(6 / (fan_in + fan_out))
    """

    def __init__(self, gain=1.):
        self.gain = gain

    def __call__(self, feature=None, weight=None):
        if feature != None:
            embed = torch.nn.Embedding(feature.vocab_size, feature.embed_dim)
            weight = embed.weight
        if weight != None:
            torch.nn.init.xavier_uniform_(weight, self.gain)
        if feature != None:
            return embed


class Pretrained(object):
    """Creates Embedding instance from given 2-dimensional FloatTensor.

    Args:
        embedding (Tensor): FloatTensor containing weights for the Embedding.
                First dimension is being passed to Embedding as ``num_embeddings``, second as ``embedding_dim``.
        freeze (boolean, optional): If ``True``, the tensor does not get updated in the learning process.
    """

    def __init__(self, embedding, freeze=True):
        self.embedding = embedding
        self.freeze = freeze

    def __call__(self, feature):
        assert feature.vocab_size == self.embedding.shape[0] and feature.embed_dim == self.embedding.shape[1]
        embed = torch.nn.Embedding.from_pretrained(self.embedding, freeze=self.freeze)
        return embed
