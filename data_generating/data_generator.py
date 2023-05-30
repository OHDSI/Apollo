import random
from collections import ChainMap, defaultdict
from typing import Iterator, Dict

import numpy as np
import pandas as pd

from data_generating.learning_objective import LearningObjective
from data_generating.parquet_data_iterator import ParquetDataIterator


SEQUENCE_LENGTH_COLUMN_NAME = "num_of_concepts"


class DataGenerator:
    """
    Generate data for tensorflow from parquet files. Iterates over the sequence data created using the CDM
    processing. To be used with tf.data.Dataset.from_generator
    """

    def __init__(self,
                 training_data_path: str,
                 batch_size: int,
                 max_sequence_length: int,
                 min_sequence_length: int,
                 is_training: bool,
                 learning_objectives: list[LearningObjective] = None):
        """
        Initialization

        Args:
            training_data_path: Path to the folder containing the parquet files.
            batch_size: Number of examples in a batch.
            max_sequence_length: The maximum length of a sequence.
            min_sequence_length: The minimum length of a sequence. Persons with sequences shorter than this will be
                ignored.
            is_training: If true, the generated data is intended for training, and for example subsequences will be
                sampled and tokens will be masked.
            learning_objectives: One or more learning objectives for which to generate data.
        """
        self._parquet_path = training_data_path
        self._batch_size = batch_size
        self._max_sequence_length = max_sequence_length
        self._min_sequence_length = min_sequence_length
        self._is_training = is_training
        self._parquet_data_iterator = ParquetDataIterator(training_data_path)
        self._nrows = len(self._parquet_data_iterator)
        self._learning_objectives = learning_objectives
        for learning_objective in self._learning_objectives:
            learning_objective.initialize(parquet_data_iterator=self._parquet_data_iterator,
                                          max_sequence_length=self._max_sequence_length,
                                          is_training=self._is_training)

    def __len__(self) -> int:
        """The number of batches per epoch"""
        return int(np.ceil(self._nrows / self._batch_size))

    def generator(self) -> Iterator[tuple[Dict, Dict]]:
        """Generate data for tensorflow"""
        while True:
            try:
                yield self._get_batch()
            except StopIteration:
                break

    def _get_batch(self) -> tuple[Dict, Dict]:
        """Get a batch of data for tensorflow"""
        row_batch = self._get_row_batch()
        input_dicts = defaultdict(list)
        output_dicts = defaultdict(list)
        for row in row_batch:
            begin_index, end_index = self._create_begin_end_indices(row)
            for learning_objective in self._learning_objectives:
                inputs, outputs = learning_objective.process_row(row, begin_index, end_index)
                for key, value in inputs.items():
                    input_dicts[key].append(value)
                for key, value in outputs.items():
                    output_dicts[key].append(value)
        return dict(input_dicts), dict(output_dicts)

    def _get_row_batch(self) -> list[pd.DataFrame]:
        """Get a batch of rows"""
        row_batch = []
        while len(row_batch) < self._batch_size:
            try:
                row = next(self._parquet_data_iterator)
            except StopIteration:
                break
            if row[SEQUENCE_LENGTH_COLUMN_NAME] >= self._min_sequence_length:
                row_batch.append(row)
        if len(row_batch) == 0:
            raise StopIteration
        return row_batch

    def _create_begin_end_indices(self, row: pd.DataFrame) -> tuple[int, int]:
        """Create begin and end indices for a row, either by sampling a sequence or using the whole sequence"""
        seq_length = row[SEQUENCE_LENGTH_COLUMN_NAME]
        if self._is_training:
            cursor = random.randint(0, seq_length - 1)
            half_window_size = int(self._max_sequence_length / 2)
            start_index = max(0, cursor - half_window_size)
            end_index = min(cursor + half_window_size, seq_length)
            if start_index < end_index:
                return start_index, end_index
            else:
                return 0, seq_length
        else:
            return 0, seq_length

    def get_tf_dataset_schema(self):
        """
        Combine the input and output tensorflow data schema from multiple learning objectives
        :return: A tuple of input and output tensorflow data schema.
        """
        input_dict_schemas = []
        output_dict_schemas = []
        for learning_objective in self._learning_objectives:
            input_dict_schema, output_dict_schema = learning_objective.get_tf_dataset_schema()
            input_dict_schemas.append(input_dict_schema)
            output_dict_schemas.append(output_dict_schema)
        return dict(ChainMap(*input_dict_schemas)), dict(ChainMap(*output_dict_schemas))
