import torch


class RandomNormal(object):
    """Returns an embedding initialized with a normal distribution.

    Args:
        mean (float): the mean of the normal distribution
        std (float): the standard deviation of the normal distribution
    """

    def __init__(self, mean=0.0, std=1.0):
        self.mean = mean
        self.std = std

    def __call__(self, vocab_size, embed_dim):
        embed = torch.nn.Embedding(vocab_size, embed_dim)
        torch.nn.init.normal_(embed.weight, self.mean, self.std)
        return embed


class RandomUniform(object):
    """Returns an embedding initialized with a uniform distribution.

    Args:
        minval (float): Lower bound of the range of random values of the uniform distribution.
        maxval (float): Upper bound of the range of random values of the uniform distribution.
    """

    def __init__(self, minval=0.0, maxval=1.0):
        self.minval = minval
        self.maxval = maxval

    def __call__(self, vocab_size, embed_dim):
        embed = torch.nn.Embedding(vocab_size, embed_dim)
        torch.nn.init.uniform_(embed.weight, self.minval, self.maxval)
        return embed


class XavierNormal(object):
    """Returns an embedding initialized with  the method described in
    `Understanding the difficulty of training deep feedforward neural networks`
    - Glorot, X. & Bengio, Y. (2010), using a uniform distribution.

    Args:
        gain (float): stddev = gain*sqrt(2 / (fan_in + fan_out))
    """

    def __init__(self, gain=1.0):
        self.gain = gain

    def __call__(self, vocab_size, embed_dim):
        embed = torch.nn.Embedding(vocab_size, embed_dim)
        torch.nn.init.xavier_normal_(embed.weight, self.gain)
        return embed


class XavierUniform(object):
    """Returns an embedding initialized with the method described in
    `Understanding the difficulty of training deep feedforward neural networks`
    - Glorot, X. & Bengio, Y. (2010), using a uniform distribution.

    Args:
        gain (float): stddev = gain*sqrt(6 / (fan_in + fan_out))
    """

    def __init__(self, gain=1.0):
        self.gain = gain

    def __call__(self, vocab_size, embed_dim):
        embed = torch.nn.Embedding(vocab_size, embed_dim)
        torch.nn.init.xavier_uniform_(embed.weight, self.gain)
        return embed


class Pretrained(object):
    """Creates Embedding instance from given 2-dimensional FloatTensor.

    Args:
        embedding_weight(Tensor or ndarray or List[List[int]]): FloatTensor containing weights for the Embedding.
                First dimension is being passed to Embedding as ``num_embeddings``, second as ``embedding_dim``.
        freeze (boolean, optional): If ``True``, the tensor does not get updated in the learning process.
    """

    def __init__(self, embedding_weight, freeze=True):
        self.embedding_weight = torch.FloatTensor(embedding_weight)
        self.freeze = freeze

    def __call__(self, vocab_size, embed_dim):
        assert vocab_size == self.embedding_weight.shape[0] and embed_dim == self.embedding_weight.shape[1]
        embed = torch.nn.Embedding.from_pretrained(self.embedding_weight, freeze=self.freeze)
        return embed
