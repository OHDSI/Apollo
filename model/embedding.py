import math
from typing import Dict

import torch
from torch import nn, Tensor

# Inspired by https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py#L180
# and https://pytorch.org/tutorials/beginner/transformer_tutorial.html


class PositionalEmbedding(torch.nn.Module):

    def __init__(self, num_embeddings: int, embedding_dim: int):
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

    def __init__(self, out_features: int):
        super().__init__()
        self.out_features = out_features
        self.w0 = nn.parameter.Parameter(torch.randn(1, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(1))
        self.w = nn.parameter.Parameter(torch.randn(1, out_features-1))
        self.b = nn.parameter.Parameter(torch.randn(out_features-1))
        self.f = torch.sin

    def forward(self, tau):
        part0 = (self.w0 * tau) + self.b0
        # Probably a nicer way to do this using broadcasting:
        part1 = self.f(tau.unsqueeze(-1).expand(-1,-1, self.out_features - 1) *
                       self.w.unsqueeze(0).expand(tau.shape[0], tau.shape[1], -1))
        return torch.cat([part0.unsqueeze(-1), part1], -1)
