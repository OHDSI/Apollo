import math
from typing import Dict

from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from data_loading.tokenizer import ConceptTokenizer
from data_loading.variable_names import ModelInputNames
from model.embedding import PositionalEmbedding
from training.train_settings import TrainingSettings

# Inspired by https://pytorch.org/tutorials/beginner/transformer_tutorial.html
# and https://coaxsoft.com/blog/building-bert-with-pytorch-from-scratch


class TransformerModel(nn.Module):

    def __init__(self, settings: TrainingSettings, tokenizer: ConceptTokenizer):
        super().__init__()
        self.model_type = 'Transformer'

        # Embeddings:
        self.token_embeddings = nn.Embedding(num_embeddings=tokenizer.get_vocab_size(),
                                             embedding_dim=settings.hidden_size,
                                             padding_idx=tokenizer.get_padding_token_id())
        nn.init.xavier_uniform_(self.token_embeddings.weight)
        self.position_embeddings = PositionalEmbedding(num_embeddings=settings.max_sequence_length,
                                                       embedding_dim=settings.hidden_size)
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

        # Decoder:
        self.masked_token_decoder = nn.Linear(in_features=settings.hidden_size,
                                              out_features=tokenizer.get_vocab_size())
        self.masked_token_decoder.bias.data.zero_()
        # decoder is shared with embedding layer
        self.masked_token_decoder.weight = self.token_embeddings.weight

    def forward(
            self,
            inputs: Dict[str, Tensor]
    ) -> Tensor:
        masked_token_ids = inputs[ModelInputNames.MASKED_TOKEN_IDS]
        visit_concept_orders = inputs[ModelInputNames.VISIT_CONCEPT_ORDERS]
        padding_mask = inputs[ModelInputNames.PADDING_MASK]

        # Not sure about the sqrt here, but it's in multiple BERT implementations:
        inputs_embeds = self.token_embeddings(masked_token_ids) * math.sqrt(self.token_embeddings.embedding_dim)
        position_embeddings = self.position_embeddings(visit_concept_orders)
        embeddings = inputs_embeds + position_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)

        encoded = self.transformer_encoder(src=embeddings, src_key_padding_mask=padding_mask)

        token_predictions = self.masked_token_decoder(encoded) # No softmax here, as it's included in CrossEntropyLoss
        return token_predictions
        # first_word = encoded[:, 0, :]
        # return self.softmax(token_predictions), self.classification_layer(first_word)
