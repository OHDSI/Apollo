import random
from typing import List, Dict, Union

import numpy as np

from data_loading.learning_objectives import (LearningObjective,
                                              NextTokenLearningObjective,
                                              NextVisitConceptsLearningObjective)
from data_loading.model_inputs import InputTransformer, find_last_index
from data_loading.variable_names import DataNames

TRUNCATE_TYPES = ["random", "tail"]


class ApolloDataTransformer:
    def __init__(self,
                 learning_objectives: List[LearningObjective],
                 input_transformer: InputTransformer,
                 max_sequence_length: int = 512,
                 truncate_type: str = "random"):
        """
        Initialization
        Args:
            learning_objectives: A list of learning objectives for which to generate data.
            max_sequence_length: The maximum length of a sequence.
            truncate_type: The type of truncation to use. Either "random" or "tail". If "tail", the last
                max_sequence_length tokens will be used.
        """
        if truncate_type not in TRUNCATE_TYPES:
            raise ValueError(f"Unknown truncate type: {truncate_type}. Must be one of {TRUNCATE_TYPES}")
        self._learning_objectives = learning_objectives
        self._keep_next_token = False
        for learning_objective in learning_objectives:
            if isinstance(learning_objective, NextTokenLearningObjective):
                self._keep_next_token = True
        self._keep_next_visit = False
        for learning_objective in learning_objectives:
            if isinstance(learning_objective, NextVisitConceptsLearningObjective):
                self._keep_next_visit = True
        if self._keep_next_token and self._keep_next_visit:
            raise ValueError("Cannot have both next token and next visit learning objectives")
        self._input_transformer = input_transformer
        self._max_sequence_length = max_sequence_length
        self._truncate_type = truncate_type

    def transform(self, row: Dict) -> Dict[str, Union[np.ndarray, float]]:
        """
        Transform the data into the format required by the learning objectives.
        Args:
            row: A row of data from the data generator.

        Returns:
            A dictionary containing the inputs to the model.
        """
        begin_index, end_index = self._create_begin_end_indices(row)
        all_inputs = self._input_transformer.process_row(row, begin_index, end_index, self._max_sequence_length)
        for learning_objective in self._learning_objectives:
            inputs = learning_objective.process_row(row, begin_index, end_index, self._max_sequence_length)
            all_inputs.update(inputs)
        return all_inputs

    def _create_begin_end_indices(self, row: Dict) -> tuple[int, int]:
        """
        Create start and end indices for a row, either by sampling a sequence or using the whole sequence. The start
        and end indices imply a sequence of length of at most max_sequence_length-1, since the [CLS] token is added
        to the beginning of the sequence.

        Args:
            row: A row of data from the data generator.

        Returns:
            A tuple of begin and end indices.
        """
        seq_length = row[DataNames.SEQUENCE_LENGTH]
        new_max_length = self._max_sequence_length - 1  # Subtract one for the [CLS] token
        if self._keep_next_visit:
            last_visit = row[DataNames.VISIT_CONCEPT_ORDERS][-1]
            max_index = find_last_index(row[DataNames.VISIT_CONCEPT_ORDERS], last_visit - 1)
            if max_index == -1:
                return 0, 0
            if max_index + 1 > new_max_length and self._truncate_type == "random":
                start_index = random.randint(0, max_index + 1 - new_max_length)
                end_index = min(max_index, start_index + new_max_length)
                return start_index, end_index
            else:
                start_index = max(0, max_index + 1 - new_max_length)
                return start_index, max_index
        else:
            next_token_offset = 1 if self._keep_next_token else 0
            if seq_length > new_max_length and self._truncate_type == "random":
                start_index = random.randint(0, seq_length - new_max_length - next_token_offset)
                end_index = min(seq_length - next_token_offset, start_index + new_max_length)
                return start_index, end_index
            else:
                return max(0, seq_length - new_max_length - next_token_offset), seq_length - next_token_offset
