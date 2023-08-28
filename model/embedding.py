import math
from typing import Dict

import torch
from torch import nn, Tensor

from training.train_settings import TrainingSettings
from data_loading.tokenizer import ConceptTokenizer
from data_loading.variable_names import ModelInputNames

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

