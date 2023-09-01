import random
from typing import List, Dict, Literal

from data_loading.learning_objective import LearningObjective


SEQUENCE_LENGTH_COLUMN_NAME = "num_of_concepts"
TRUNCATE_TYPE = Literal["random", "last"]


class ApolloDataTransformer:
    def __init__(self,
                 learning_objectives: List[LearningObjective],
                 max_sequence_length: int = 512,
                 truncate_type: TRUNCATE_TYPE = "random"):
        """
        Initialization
        Args:
            learning_objectives: A list of learning objectives for which to generate data.
            max_sequence_length: The maximum length of a sequence.
            truncate_type: The type of truncation to use. Either "random" or "last". If "last", the last
                max_sequence_length tokens will be used.
        """
        self._learning_objectives = learning_objectives
        self._max_sequence_length = max_sequence_length
        self._truncate_type = truncate_type

    def transform(self, row: Dict) -> tuple[Dict, Dict]:
        """
        Transform the data into the format required by the learning objectives.
        Args:
            row: A row of data from the data generator.

        Returns:
            A tuple of dictionaries. The first dictionary contains the inputs to the model, the second dictionary
            contains the outputs of the model.
        """
        begin_index, end_index = self._create_begin_end_indices(row)
        all_inputs = {}
        all_outputs = {}
        for learning_objective in self._learning_objectives:
            inputs, outputs = learning_objective.process_row(row, begin_index, end_index, self._max_sequence_length)
            all_inputs.update(inputs)
            all_outputs.update(outputs)
        return all_inputs, all_outputs

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
        seq_length = row[SEQUENCE_LENGTH_COLUMN_NAME]
        new_max_length = self._max_sequence_length - 1  # Subtract one for the [CLS] token
        if seq_length > new_max_length and self._truncate_type == "random":
            # Note: to match most likely use cases, we should probably sample to end at end of a vists
            start_index = random.randint(0, seq_length - new_max_length)
            end_index = min(seq_length, start_index + new_max_length)
            return start_index, end_index
        else:
            return max(0, seq_length - new_max_length), seq_length
