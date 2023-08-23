import math
from typing import Dict

import torch
from torch import nn, Tensor

from training.train_settings import CehrBertSettings
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


class JointEmbedding(nn.Module):
    """Construct the embeddings from word, and position embeddings."""

    def __init__(self, settings: CehrBertSettings, tokenizer: ConceptTokenizer):
        super().__init__()
        self.word_embeddings = nn.Embedding(num_embeddings=tokenizer.get_vocab_size(),
                                            embedding_dim=settings.hidden_size,
                                            padding_idx=tokenizer.get_padding_token_id())
        self.position_embeddings = PositionalEmbedding(num_embeddings=settings.max_sequence_length,
                                                       embedding_dim=settings.hidden_size)
        self.layer_norm = nn.LayerNorm(normalized_shape=settings.hidden_size)
        self.dropout = nn.Dropout(settings.hidden_dropout_prob)

    def forward(
        self,
        inputs: Dict[str, Tensor]
    ) -> Tensor:
        masked_token_ids = inputs[ModelInputNames.MASKED_TOKEN_IDS]
        visit_concept_orders = inputs[ModelInputNames.VISIT_CONCEPT_ORDERS]
        # Not sure about the sqrt here, but it's in multiple BERT implementations:
        inputs_embeds = self.word_embeddings(masked_token_ids) * math.sqrt(self.word_embeddings.embedding_dim)
        position_embeddings = self.position_embeddings(visit_concept_orders)
        embeddings = inputs_embeds + position_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
