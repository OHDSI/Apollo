import math
from typing import Dict

import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from data_loading.tokenizer import ConceptTokenizer
from data_loading.variable_names import ModelInputNames, ModelOutputNames
from model.embedding import PositionalEmbedding, TimeEmbedding
from training.train_settings import TrainingSettings


class TransformerModel(nn.Module):

    def __init__(self,
                 settings: TrainingSettings,
                 tokenizer: ConceptTokenizer,
                 visit_tokenizer: ConceptTokenizer):
        super().__init__()
        self.settings = settings

        # Embeddings:
        self.token_embeddings = nn.Embedding(num_embeddings=tokenizer.get_vocab_size(),
                                             embedding_dim=settings.hidden_size,
                                             padding_idx=tokenizer.get_padding_token_id())
        nn.init.xavier_uniform_(self.token_embeddings.weight)
        self.position_embeddings = PositionalEmbedding(num_embeddings=settings.max_sequence_length,
                                                       embedding_dim=settings.hidden_size)
        self.age_embeddings = TimeEmbedding(embedding_dim=settings.hidden_size)
        self.date_embeddings = TimeEmbedding(embedding_dim=settings.hidden_size)
        self.segment_embeddings = nn.Embedding(num_embeddings=3,
                                               embedding_dim=settings.hidden_size,
                                               padding_idx=0)
        if settings.masked_visit_concept_learning:
            self.visit_token_embeddings = nn.Embedding(num_embeddings=visit_tokenizer.get_vocab_size(),
                                                       embedding_dim=settings.hidden_size,
                                                       padding_idx=visit_tokenizer.get_padding_token_id())
            nn.init.xavier_uniform_(self.visit_token_embeddings.weight)
        self.layer_norm = nn.LayerNorm(normalized_shape=settings.hidden_size)
        self.dropout = nn.Dropout(settings.hidden_dropout_prob)

        # Encoder:
        encoder_layers = TransformerEncoderLayer(d_model=settings.hidden_size,
                                                 nhead=settings.num_attention_heads,
                                                 dim_feedforward=settings.max_sequence_length,
                                                 dropout=settings.hidden_dropout_prob,
                                                 activation=settings.hidden_act,
                                                 batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layer=encoder_layers,
                                                      num_layers=settings.num_hidden_layers)
        # Seems necessary per https://github.com/pytorch/pytorch/issues/72253
        for name, param in self.transformer_encoder.named_parameters():
            if 'weight' in name and param.data.dim() == 2:
                nn.init.kaiming_uniform_(param)

        # Decoders:
        self.masked_token_decoder = nn.Linear(in_features=settings.hidden_size,
                                              out_features=tokenizer.get_vocab_size())
        self.masked_token_decoder.bias.data.zero_()
        nn.init.xavier_uniform_(self.masked_token_decoder.weight)
        # Alternatively, decoder is shared with embedding layer:
        # self.masked_token_decoder.weight = self.token_embeddings.weight
        if settings.masked_visit_concept_learning:
            self.masked_vist_token_decoder = nn.Linear(in_features=settings.hidden_size,
                                                       out_features=visit_tokenizer.get_vocab_size())
            self.masked_vist_token_decoder.bias.data.zero_()
            nn.init.xavier_uniform_(self.masked_vist_token_decoder.weight)

    def forward(
            self,
            inputs: Dict[str, Tensor]
    ) -> Dict[str, Tensor]:
        masked_token_ids = inputs[ModelInputNames.MASKED_TOKEN_IDS]
        visit_concept_orders = inputs[ModelInputNames.VISIT_CONCEPT_ORDERS]
        padding_mask = inputs[ModelInputNames.PADDING_MASK]
        ages = inputs[ModelInputNames.AGES]
        dates = inputs[ModelInputNames.DATES]
        visit_segment = inputs[ModelInputNames.VISIT_SEGMENTS]

        # Not sure about multiplication with the sqrt here, but it's in multiple BERT implementations:
        embeddings = self.token_embeddings(masked_token_ids) * math.sqrt(self.token_embeddings.embedding_dim)
        embeddings += self.age_embeddings(ages)
        embeddings += self.date_embeddings(dates)
        embeddings += self.segment_embeddings(visit_segment)
        embeddings += self.position_embeddings(visit_concept_orders)

        if self.settings.masked_visit_concept_learning:
            masked_visit_token_ids = inputs[ModelInputNames.MASKED_VISIT_TOKEN_IDS]
            embeddings += self.visit_token_embeddings(masked_visit_token_ids) * math.sqrt(
                self.visit_token_embeddings.embedding_dim)
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)

        encoded = self.transformer_encoder(src=embeddings, src_key_padding_mask=padding_mask)

        # No softmax here, as it's included in CrossEntropyLoss:
        token_predictions = self.masked_token_decoder(encoded)
        visit_token_predictions = self.masked_vist_token_decoder(encoded)
        return {ModelOutputNames.TOKEN_PREDICTIONS: token_predictions,
                ModelOutputNames.VISIT_TOKEN_PREDICTIONS: visit_token_predictions}
