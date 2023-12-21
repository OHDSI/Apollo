import math
from typing import Dict

import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from data_loading.tokenizer import ConceptTokenizer
from data_loading.variable_names import ModelInputNames, ModelOutputNames
from model.embedding import PositionalEmbedding, TimeEmbedding
from training.train_settings import ModelSettings, LearningObjectiveSettings


class TransformerModel(nn.Module):

    def __init__(self,
                 model_settings: ModelSettings,
                 learning_objective_settings: LearningObjectiveSettings,
                 tokenizer: ConceptTokenizer,
                 visit_tokenizer: ConceptTokenizer):
        super().__init__()
        self.model_settings = model_settings
        self.learning_objective_settings = learning_objective_settings
        self._frozen = False

        # Embeddings:
        embeddings_total_dim = 0
        if model_settings.concept_embedding:
            self.token_embeddings = nn.Embedding(num_embeddings=tokenizer.get_vocab_size(),
                                                 embedding_dim=model_settings.hidden_size,
                                                 padding_idx=tokenizer.get_padding_token_id())
            nn.init.xavier_uniform_(self.token_embeddings.weight)
            embeddings_total_dim += model_settings.hidden_size
        if model_settings.visit_concept_embedding:
            self.visit_concept_embeddings = nn.Embedding(num_embeddings=visit_tokenizer.get_vocab_size(),
                                                         embedding_dim=model_settings.hidden_size,
                                                         padding_idx=visit_tokenizer.get_padding_token_id())
            nn.init.xavier_uniform_(self.visit_concept_embeddings.weight)
            embeddings_total_dim += model_settings.hidden_size
        if model_settings.visit_order_embedding:
            self.visit_order_embeddings = PositionalEmbedding(num_embeddings=model_settings.max_sequence_length,
                                                              embedding_dim=model_settings.hidden_size)
            embeddings_total_dim += model_settings.hidden_size
        if model_settings.age_embedding:
            self.age_embeddings = TimeEmbedding(embedding_dim=model_settings.hidden_size)
            embeddings_total_dim += model_settings.hidden_size
        if model_settings.date_embedding:
            self.date_embeddings = TimeEmbedding(embedding_dim=model_settings.hidden_size)
            embeddings_total_dim += model_settings.hidden_size
        if model_settings.segment_embedding:
            self.segment_embeddings = nn.Embedding(num_embeddings=3,
                                                   embedding_dim=model_settings.hidden_size,
                                                   padding_idx=0)
            embeddings_total_dim += model_settings.hidden_size

        # Embedding combination:
        if model_settings.embedding_combination_method == "concat":
            self.embedding_concat_rescale_layer = nn.Linear(in_features=embeddings_total_dim,
                                                            out_features=model_settings.hidden_size)

        self.layer_norm = nn.LayerNorm(normalized_shape=model_settings.hidden_size)
        self.dropout = nn.Dropout(model_settings.hidden_dropout_prob)

        # Encoder:
        encoder_layers = TransformerEncoderLayer(d_model=model_settings.hidden_size,
                                                 nhead=model_settings.num_attention_heads,
                                                 dim_feedforward=model_settings.max_sequence_length,
                                                 dropout=model_settings.hidden_dropout_prob,
                                                 activation=model_settings.hidden_act,
                                                 batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layer=encoder_layers,
                                                      num_layers=model_settings.num_hidden_layers)
        # Seems necessary per https://github.com/pytorch/pytorch/issues/72253
        for name, param in self.transformer_encoder.named_parameters():
            if 'weight' in name and param.data.dim() == 2:
                nn.init.kaiming_uniform_(param)

        # Decoders:
        if learning_objective_settings.masked_concept_learning:
            self.masked_token_decoder = nn.Linear(in_features=model_settings.hidden_size,
                                                  out_features=tokenizer.get_vocab_size())
            self.masked_token_decoder.bias.data.zero_()
            nn.init.xavier_uniform_(self.masked_token_decoder.weight)
        if learning_objective_settings.masked_visit_concept_learning:
            self.masked_visit_token_decoder = nn.Linear(in_features=model_settings.hidden_size,
                                                        out_features=visit_tokenizer.get_vocab_size())
            self.masked_visit_token_decoder.bias.data.zero_()
            nn.init.xavier_uniform_(self.masked_visit_token_decoder.weight)
        if learning_objective_settings.next_token_prediction:
            self.next_token_decoder = nn.Linear(in_features=model_settings.hidden_size,
                                                out_features=tokenizer.get_vocab_size())
            self.next_token_decoder.bias.data.zero_()
            nn.init.xavier_uniform_(self.next_token_decoder.weight)
            self.src_mask = nn.Transformer.generate_square_subsequent_mask(model_settings.max_sequence_length)
        else:
            self.src_mask = None
        if learning_objective_settings.next_visit_concepts_prediction:
            self.next_visit_tokens_decoder = nn.Linear(in_features=model_settings.hidden_size,
                                                       out_features=tokenizer.get_vocab_size())
            self.next_visit_tokens_decoder.bias.data.zero_()
            nn.init.xavier_uniform_(self.next_visit_tokens_decoder.weight)
        if learning_objective_settings.label_prediction or learning_objective_settings.new_label_prediction:
            self.label_decoder = nn.Linear(in_features=model_settings.hidden_size,
                                           out_features=1)
            self.label_decoder.bias.data.zero_()
            nn.init.xavier_uniform_(self.label_decoder.weight)

    def forward(
            self,
            inputs: Dict[str, Tensor]
    ) -> Dict[str, Tensor]:

        embeddings = []
        if self.model_settings.concept_embedding:
            if self.learning_objective_settings.masked_concept_learning:
                token_ids = inputs[ModelInputNames.MASKED_TOKEN_IDS]
            else:
                token_ids = inputs[ModelInputNames.TOKEN_IDS]
            # Not sure about multiplication with the sqrt here, but it's in multiple BERT implementations:
            embeddings.append(self.token_embeddings(token_ids) * math.sqrt(self.token_embeddings.embedding_dim))

        if self.model_settings.visit_concept_embedding:
            if self.learning_objective_settings.masked_visit_concept_learning:
                visit_token_ids = inputs[ModelInputNames.MASKED_VISIT_TOKEN_IDS]
            else:
                visit_token_ids = inputs[ModelInputNames.VISIT_TOKEN_IDS]
            embeddings.append(
                self.visit_concept_embeddings(visit_token_ids) * math.sqrt(self.visit_concept_embeddings.embedding_dim))

        if self.model_settings.visit_order_embedding:
            embeddings.append(self.visit_order_embeddings(inputs[ModelInputNames.VISIT_CONCEPT_ORDERS]))

        if self.model_settings.age_embedding:
            embeddings.append(self.age_embeddings(inputs[ModelInputNames.AGES]))

        if self.model_settings.date_embedding:
            embeddings.append(self.date_embeddings(inputs[ModelInputNames.DATES]))

        if self.model_settings.segment_embedding:
            embeddings.append(self.segment_embeddings(inputs[ModelInputNames.VISIT_SEGMENTS]))

        if self.model_settings.embedding_combination_method == "concat":
            embeddings = torch.cat(embeddings, dim=-1)
            embeddings = self.embedding_concat_rescale_layer(embeddings)
            embeddings = torch.tanh(embeddings)
        elif self.model_settings.embedding_combination_method == "sum":
            embeddings = torch.stack(embeddings, dim=0).sum(dim=0)

        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)

        if self.learning_objective_settings.next_token_prediction:
            encoded = self.transformer_encoder(src=embeddings,
                                               mask=self.src_mask,
                                               src_key_padding_mask=inputs[ModelInputNames.PADDING_MASK],
                                               is_causal=True)
        else:
            encoded = self.transformer_encoder(src=embeddings, src_key_padding_mask=inputs[ModelInputNames.PADDING_MASK])

        predictions = {}
        if self.learning_objective_settings.masked_concept_learning:
            # No softmax here, as it's included in CrossEntropyLoss:
            predictions[ModelOutputNames.TOKEN_PREDICTIONS] = self.masked_token_decoder(encoded)
        if self.learning_objective_settings.masked_visit_concept_learning:
            # No softmax here, as it's included in CrossEntropyLoss:
            predictions[ModelOutputNames.VISIT_TOKEN_PREDICTIONS] = self.masked_visit_token_decoder(encoded)
        if self.learning_objective_settings.next_token_prediction:
            # No softmax here, as it's included in CrossEntropyLoss:
            predictions[ModelOutputNames.NEXT_TOKEN_PREDICTION] = self.next_token_decoder(encoded)
        if self.learning_objective_settings.next_visit_concepts_prediction:
            # No softmax here, as it's included in CrossEntropyLoss:
            predictions[ModelOutputNames.NEXT_VISIT_TOKENS_PREDICTION] = self.next_visit_tokens_decoder(encoded[:, 0, :])
        if self.learning_objective_settings.label_prediction or self.learning_objective_settings.new_label_prediction:
            predictions[ModelOutputNames.LABEL_PREDICTIONS] = torch.sigmoid(self.label_decoder(encoded[:, 0, :]))
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

    def to(self, device: torch.device):
        super().to(device)
        if self.src_mask is not None:
            self.src_mask = self.src_mask.to(device)
        return self

