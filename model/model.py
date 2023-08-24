from typing import Dict

from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from data_loading.tokenizer import ConceptTokenizer
from data_loading.variable_names import ModelInputNames
from model.embedding import JointEmbedding
from training.train_settings import TrainingSettings

# Inspired by https://pytorch.org/tutorials/beginner/transformer_tutorial.html
# and https://coaxsoft.com/blog/building-bert-with-pytorch-from-scratch


class TransformerModel(nn.Module):

    def __init__(self, settings: TrainingSettings, tokenizer: ConceptTokenizer):
        super().__init__()
        self.model_type = 'Transformer'
        self.joint_embedding = JointEmbedding(settings, tokenizer)
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
        # Note: HF uses bias per token, may want to consider that here:
        # https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py#L693
        self.token_prediction_layer = nn.Linear(in_features=settings.hidden_size,
                                                out_features=tokenizer.get_vocab_size())
        nn.init.xavier_uniform_(self.token_prediction_layer.weight)
        self.softmax = nn.LogSoftmax(dim=-1)
        # self.classification_layer = nn.Linear(dim_inp, 2)

    def forward(
            self,
            inputs: Dict[str, Tensor]
    ) -> Tensor:
        embedded = self.joint_embedding(inputs)
        padding_mask = inputs[ModelInputNames.PADDING_MASK]
        encoded = self.transformer_encoder(src=embedded, src_key_padding_mask=padding_mask)
        token_predictions = self.token_prediction_layer(encoded)
        return self.softmax(token_predictions)
        # first_word = encoded[:, 0, :]
        # return self.softmax(token_predictions), self.classification_layer(first_word)
