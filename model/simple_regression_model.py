"""
A simple regression model that can be used as baseline for the transformer model. It simply takes the input tokens,
and turns them into a single vector representation, which is then used to predict the label.
"""

from typing import Dict

import torch
from torch import nn, Tensor

from data_loading.tokenizer import ConceptTokenizer
from data_loading.variable_names import ModelInputNames, ModelOutputNames
from model.model_settings import ModelSettings
from training.train_settings import LearningObjectiveSettings


class SimpleRegressionModel(torch.nn.Module):
    def __init__(self,
                 model_settings: ModelSettings,
                 learning_objective_settings: LearningObjectiveSettings,
                 tokenizer: ConceptTokenizer,
                 visit_tokenizer: ConceptTokenizer):
        super().__init__()
        self.model_settings = model_settings
        self.learning_objective_settings = learning_objective_settings
        self._frozen = False
        self._vocab_size = tokenizer.get_vocab_size()

        # Decoders:
        if learning_objective_settings.masked_concept_learning or learning_objective_settings.masked_visit_concept_learning:
            raise NotImplementedError("Masked concept learning and masked visit learning are not implemented for the "
                                      "simple regression model.")
        if learning_objective_settings.label_prediction:
            self.label_decoder = nn.Linear(in_features=tokenizer.get_vocab_size(),
                                           out_features=1)
            self.label_decoder.bias.data.zero_()
            nn.init.xavier_uniform_(self.label_decoder.weight)

    def forward(
            self,
            inputs: Dict[str, Tensor]
    ) -> Dict[str, Tensor]:

        if self.learning_objective_settings.masked_concept_learning:
            token_ids = inputs[ModelInputNames.MASKED_TOKEN_IDS]
        else:
            token_ids = inputs[ModelInputNames.TOKEN_IDS]
        input_vector = torch.zeros((token_ids.shape[0], self._vocab_size), device=token_ids.device)
        input_vector.scatter_(1, token_ids, 1)

        predictions = {}
        if self.learning_objective_settings.label_prediction:
            predictions[ModelOutputNames.LABEL_PREDICTIONS] = torch.sigmoid(self.label_decoder(input_vector))
        return predictions

    def freeze_non_head(self):
        self._frozen = True

    def unfreeze_all(self):
        self._frozen = False

    def is_frozen(self):
        return self._frozen
