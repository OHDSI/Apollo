import math

import torch
from torch import nn, Tensor


class PositionalEmbedding(torch.nn.Module):

    def __init__(self, num_embeddings: int, embedding_dim: int):
        """
        Initialization.
        Args:
            num_embeddings: Number of embeddings.
            embedding_dim: Number of dimensions of the embedding. Must be even.
        """
        super().__init__()
        position = torch.arange(num_embeddings).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2) * (-math.log(10000.0) / embedding_dim))
        pe = torch.zeros(num_embeddings, embedding_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        return self.pe[x]


class TimeEmbedding(nn.Module):
    """
    Time2Vec embedding layer as described in https://arxiv.org/pdf/1907.05321.pdf
    """

    def __init__(self, embedding_dim: int):
        """
        Initialization.
        Args:
            embedding_dim: Number of dimensions of the embedding. Must be at least 2.
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.w0 = nn.parameter.Parameter(torch.randn(1, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(1))
        self.w = nn.parameter.Parameter(torch.randn(1, embedding_dim-1))
        self.b = nn.parameter.Parameter(torch.randn(embedding_dim-1))
        self.f = torch.sin

    def forward(self, tau: Tensor) -> Tensor:
        part0 = (self.w0 * tau) + self.b0
        # Probably a nicer way to do this using broadcasting:
        part1 = self.f(tau.unsqueeze(-1).expand(-1, -1, self.embedding_dim - 1) *
                       self.w.unsqueeze(0).expand(tau.shape[0], tau.shape[1], -1) +
                       self.b.unsqueeze(0).expand(tau.shape[0], tau.shape[1], -1))
        return torch.cat([part0.unsqueeze(-1), part1], -1)
