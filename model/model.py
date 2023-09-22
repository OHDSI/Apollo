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
        self._frozen = False

        # Embeddings:
        embeddings_total_dim = 0
        if settings.masked_concept_learning or settings.label_prediction:
            self.token_embeddings = nn.Embedding(num_embeddings=tokenizer.get_vocab_size(),
                                                 embedding_dim=settings.hidden_size,
                                                 padding_idx=tokenizer.get_padding_token_id())
            nn.init.xavier_uniform_(self.token_embeddings.weight)
            embeddings_total_dim += settings.hidden_size
            self.position_embeddings = PositionalEmbedding(num_embeddings=settings.max_sequence_length,
                                                           embedding_dim=settings.hidden_size)
            embeddings_total_dim += settings.hidden_size
            self.age_embeddings = TimeEmbedding(embedding_dim=settings.hidden_size)
            embeddings_total_dim += settings.hidden_size
            self.date_embeddings = TimeEmbedding(embedding_dim=settings.hidden_size)
            embeddings_total_dim += settings.hidden_size
            self.segment_embeddings = nn.Embedding(num_embeddings=3,
                                                   embedding_dim=settings.hidden_size,
                                                   padding_idx=0)
            embeddings_total_dim += settings.hidden_size
        if settings.masked_visit_concept_learning or settings.label_prediction:
            self.visit_token_embeddings = nn.Embedding(num_embeddings=visit_tokenizer.get_vocab_size(),
                                                       embedding_dim=settings.hidden_size,
                                                       padding_idx=visit_tokenizer.get_padding_token_id())
            nn.init.xavier_uniform_(self.visit_token_embeddings.weight)
            embeddings_total_dim += settings.hidden_size
        if settings.embedding_combination_method == "concat":
            self.embedding_concat_rescale_layer = nn.Linear(in_features=embeddings_total_dim,
                                                            out_features=settings.hidden_size)

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
        if settings.masked_concept_learning:
            self.masked_token_decoder = nn.Linear(in_features=settings.hidden_size,
                                                  out_features=tokenizer.get_vocab_size())
            self.masked_token_decoder.bias.data.zero_()
            nn.init.xavier_uniform_(self.masked_token_decoder.weight)
            # Alternatively, decoder is shared with embedding layer:
            # self.masked_token_decoder.weight = self.token_embeddings.weight
        if settings.masked_visit_concept_learning:
            self.masked_visit_token_decoder = nn.Linear(in_features=settings.hidden_size,
                                                        out_features=visit_tokenizer.get_vocab_size())
            self.masked_visit_token_decoder.bias.data.zero_()
            nn.init.xavier_uniform_(self.masked_visit_token_decoder.weight)
        if settings.label_prediction:
            self.label_decoder = nn.Linear(in_features=settings.hidden_size,
                                           out_features=2)
            self.label_decoder.bias.data.zero_()
            nn.init.xavier_uniform_(self.label_decoder.weight)

    def forward(
            self,
            inputs: Dict[str, Tensor]
    ) -> Dict[str, Tensor]:

        embeddings = []
        if self.settings.masked_concept_learning or self.settings.label_prediction:
            if self.settings.masked_concept_learning:
                token_ids = inputs[ModelInputNames.MASKED_TOKEN_IDS]
            else:
                token_ids = inputs[ModelInputNames.TOKEN_IDS]
            # Not sure about multiplication with the sqrt here, but it's in multiple BERT implementations:
            embeddings.append(self.token_embeddings(token_ids) * math.sqrt(self.token_embeddings.embedding_dim))
            embeddings.append(self.age_embeddings(inputs[ModelInputNames.AGES]))
            embeddings.append(self.date_embeddings(inputs[ModelInputNames.DATES]))
            embeddings.append(self.segment_embeddings(inputs[ModelInputNames.VISIT_SEGMENTS]))
            embeddings.append(self.position_embeddings(inputs[ModelInputNames.VISIT_CONCEPT_ORDERS]))
        if self.settings.masked_visit_concept_learning:
            masked_visit_token_ids = inputs[ModelInputNames.MASKED_VISIT_TOKEN_IDS]
            visit_embeddings = self.visit_token_embeddings(masked_visit_token_ids) * math.sqrt(
                self.visit_token_embeddings.embedding_dim)
            embeddings.append(visit_embeddings)

        if self.settings.embedding_combination_method == "concat":
            embeddings = torch.cat(embeddings, dim=-1)
            embeddings = self.embedding_concat_rescale_layer(embeddings)
            embeddings = torch.tanh(embeddings)
        elif self.settings.embedding_combination_method == "sum":
            embeddings = torch.stack(embeddings, dim=0).sum(dim=0)

        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)

        encoded = self.transformer_encoder(src=embeddings, src_key_padding_mask=inputs[ModelInputNames.PADDING_MASK])

        predictions = {}
        if self.settings.masked_concept_learning:
            # No softmax here, as it's included in CrossEntropyLoss:
            predictions[ModelOutputNames.TOKEN_PREDICTIONS] = self.masked_token_decoder(encoded)
        if self.settings.masked_visit_concept_learning:
            # No softmax here, as it's included in CrossEntropyLoss:
            predictions[ModelOutputNames.VISIT_TOKEN_PREDICTIONS] = self.masked_visit_token_decoder(encoded)
        if self.settings.label_prediction:
            predictions[ModelOutputNames.LABEL_PREDICTIONS] = self.label_decoder(encoded[:, 0, :])
        return predictions

    def freeze_non_head(self):
        """Freeze all parameters except the head layers."""
        for name, param in self.named_parameters():
            if 'masked_token_decoder' in name or 'masked_visit_token_decoder' in name or 'label_decoder' in name:
                continue
            param.requires_grad = False
        self._frozen = True

    def unfreeze_all(self):
        """Unfreeze all parameters."""
        for param in self.parameters():
            param.requires_grad = True
        self._frozen = False

    def is_frozen(self):
        return self._frozen
