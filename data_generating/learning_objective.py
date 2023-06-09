import os
import random
from abc import ABC, abstractmethod
from typing import Dict

import numpy as np
import pandas as pd
import tensorflow as tf

from data_generating import tokenizer
from data_generating.abstract_data_generator import AbstractDataGenerator


class LayerInputNames:
    """
    Names of the inputs to the model. These inputs are generated by the data generator using the learning objectives.
    """
    LABEL = "label"
    MASKED_VISIT_CONCEPTS = "masked_visit_concepts"
    MASK_VISIT = "mask_visit"
    VISIT_PREDICTIONS = "visit_predictions"
    MASKED_CONCEPT_IDS = "masked_concept_ids"
    CONCEPT_IDS = "concept_ids"
    MASK = "mask"
    TIME_STAMPS = "time_stamps"
    VISIT_SEGMENTS = "visit_segments"
    AGES = "ages"
    VISIT_CONCEPT_ORDERS = "visit_concept_orders"
    CONCEPT_PREDICTIONS = "concept_predictions"


def _pad_sequence(sequence: np.ndarray[any], padding_value: any, max_sequence_length: int) -> np.ndarray[any]:
    """
    Pad a sequence to a given length.

    Args
        sequence: The sequence to pad.
        max_sequence_length: The length to pad to.
        adding_value: The value to pad with.
    Returns
        The padded sequence.
    """
    n_to_pad = max_sequence_length - len(sequence)
    if n_to_pad > 0:
        sequence = np.append(sequence, [padding_value] * n_to_pad)
    return sequence


class LearningObjective(ABC):
    """
    A learning objective is a task that can be learned from the data. For example, predicting the next visit. This
    class is used to generate the data for the learning objective.
    """

    @abstractmethod
    def initialize(self, data_generator: AbstractDataGenerator):
        """
        An initializer called by the DataGenerator. Other, objective-specific initialization, can be done in __init__.

        Args
            data_generator: The calling data generator.
        """
        pass

    @abstractmethod
    def process_row(self, row: pd.DataFrame, start_index: int, end_index: int) -> tuple[Dict, Dict]:
        """
        Process a row to generate input and output data for the learning objective

        Args
            row: The row to process, as generated by the CDM processing.
            start_index: Any sequence in the row should start at this index.
            end_index: Any sequence in the row should end at this index.

        Returns
            Two dictonaries to be used by Tensorflow. The first is the input, the second is the output.
        """
        pass

    @abstractmethod
    def get_tf_dataset_schema(self) -> tuple[Dict, Dict]:
        """
        Get the schema for the input and output to the tensorflow Dataset

        Returns
            A tuple of two dictionaries. The first is the input schema, the second is the output schema.
        """
        pass


class BertFineTuningLearningObjective(LearningObjective):
    def initialize(self, data_generator: AbstractDataGenerator):
        pass

    def get_tf_dataset_schema(self) -> tuple[Dict, Dict]:
        output_dict_schema = {LayerInputNames.LABEL: tf.int32}
        return {}, output_dict_schema

    def process_row(self, row: pd.DataFrame, start_index: int, end_index: int) -> tuple[Dict, Dict]:
        output_dict = {LayerInputNames.LABEL: row.label}
        return {}, output_dict


class VisitPredictionLearningObjective(LearningObjective):

    def __init__(self, work_folder: str, reuse_tokenizer: bool = True):
        """
        Initialization
        Args:
            work_folder: The folder where the tokenizer will be saved.
            reuse_tokenizer: If true, the tokenizer will be loaded from the work_folder if it exists.
        """
        self._work_folder = work_folder
        self._visit_tokenizer = None
        self._reuse_tokenizer = reuse_tokenizer
        self._max_sequence_length = None

    def initialize(self, data_generator: AbstractDataGenerator):
        json_file = os.path.join(self._work_folder, "_visit_tokenizer.json")
        if self._reuse_tokenizer and os.path.exists(json_file):
            self._visit_tokenizer = tokenizer.load_from_json(json_file)
        else:
            self._visit_tokenizer = tokenizer.ConceptTokenizer()
            self._visit_tokenizer.fit_on_concept_sequences(data_generator.get_parquet_data_iterator(),
                                                           "visit_concept_ids")
            self._visit_tokenizer.save_to_json(json_file)
        self._max_sequence_length = data_generator.get_max_sequence_length()

    def get_tf_dataset_schema(self):
        input_dict_schema = {
            LayerInputNames.MASKED_VISIT_CONCEPTS: tf.int32,
            LayerInputNames.MASK_VISIT: tf.int32
        }
        output_dict_schema = {LayerInputNames.VISIT_PREDICTIONS: tf.int32}
        return input_dict_schema, output_dict_schema

    def process_row(self, row: pd.DataFrame, start_index: int, end_index: int) -> tuple[Dict, Dict]:
        visit_concept_ids = row.visit_concept_ids[start_index:end_index]
        visit_token_ids = self._visit_tokenizer.encode(visit_concept_ids)
        masked_visit_token_ids, output_mask = self._mask_visit_concepts(visit_token_ids)
        masked_visit_token_ids = _pad_sequence(sequence=masked_visit_token_ids,
                                               padding_value=self._visit_tokenizer.get_unused_token_id(),
                                               max_sequence_length=self._max_sequence_length)
        visit_token_ids = _pad_sequence(sequence=masked_visit_token_ids,
                                        padding_value=self._visit_tokenizer.get_unused_token_id(),
                                        max_sequence_length=self._max_sequence_length)
        visit_mask = (visit_token_ids == self._visit_tokenizer.get_unused_token_id()).astype(int)
        combined_label = np.stack([visit_token_ids, output_mask], axis=-1)
        input_dict = {
            LayerInputNames.MASKED_VISIT_CONCEPTS: masked_visit_token_ids,
            LayerInputNames.MASK_VISIT: visit_mask
        }
        output_dict = {LayerInputNames.VISIT_PREDICTIONS: combined_label}
        return input_dict, output_dict

    def _mask_visit_concepts(self, visit_concepts):
        masked_visit_concepts = np.asarray(visit_concepts).copy()
        output_mask = np.zeros((self._max_sequence_length,), dtype=int)
        for word_pos in range(0, len(visit_concepts)):
            if random.random() < 0.5:
                output_mask[word_pos] = 1
                masked_visit_concepts[word_pos] = self._visit_tokenizer.get_mask_token_id()
        return masked_visit_concepts, output_mask


class MaskedLanguageModelLearningObjective(LearningObjective):

    def __init__(self, work_folder: str, reuse_tokenizer: bool = True):
        """
        Initialization
        Args:
            work_folder: The folder where the tokenizer will be saved.
            reuse_tokenizer: If true, the tokenizer will be loaded from the work_folder if it exists.
        """
        self._work_folder = work_folder
        self._reuse_tokenizer = reuse_tokenizer
        self._concept_tokenizer = None
        self._max_sequence_length = None
        self._is_training = None

    def initialize(self, data_generator: AbstractDataGenerator):
        json_file = os.path.join(self._work_folder, "_concept_tokenizer.json")
        if self._reuse_tokenizer and os.path.exists(json_file):
            self._concept_tokenizer = tokenizer.load_from_json(json_file)
        else:
            self._concept_tokenizer = tokenizer.ConceptTokenizer()
            self._concept_tokenizer.fit_on_concept_sequences(data_generator.get_parquet_data_iterator(), "concept_ids")
            self._concept_tokenizer.save_to_json(json_file)
        self._max_sequence_length = data_generator.get_max_sequence_length()
        self._is_training = data_generator.get_is_training()

    def get_tf_dataset_schema(self):
        input_dict_schema = {
            LayerInputNames.MASKED_CONCEPT_IDS: tf.int32,
            LayerInputNames.CONCEPT_IDS: tf.int32,
            LayerInputNames.MASK: tf.int32,
            LayerInputNames.TIME_STAMPS: tf.int32,
            LayerInputNames.VISIT_SEGMENTS: tf.int32,
            LayerInputNames.AGES: tf.int32,
            LayerInputNames.VISIT_CONCEPT_ORDERS: tf.int32
        }
        output_dict_schema = {LayerInputNames.CONCEPT_PREDICTIONS: tf.int32}
        return input_dict_schema, output_dict_schema

    def process_row(self, row: pd.DataFrame, start_index: int, end_index: int) -> tuple[Dict, Dict]:
        concept_ids = row.concept_ids[start_index:end_index]
        visit_segments = row.visit_segments[start_index:end_index]
        dates = row.dates[start_index:end_index]
        ages = row.ages[start_index:end_index]
        visit_concept_orders = row.visit_concept_orders[start_index:end_index]

        token_ids = self._concept_tokenizer.encode(concept_ids)
        # Normalize the visit_orders using the smallest visit_concept_orders
        visit_concept_orders = visit_concept_orders - min(visit_concept_orders)
        masked_token_ids, output_mask = self._mask_concepts(token_ids)

        token_ids = _pad_sequence(sequence=token_ids,
                                  padding_value=self._concept_tokenizer.get_unused_token_id(),
                                  max_sequence_length=self._max_sequence_length)
        masked_token_ids = _pad_sequence(sequence=masked_token_ids,
                                         padding_value=self._concept_tokenizer.get_unused_token_id(),
                                         max_sequence_length=self._max_sequence_length)
        visit_segments = _pad_sequence(sequence=visit_segments,
                                       padding_value=self._max_sequence_length,
                                       max_sequence_length=self._max_sequence_length)
        dates = _pad_sequence(sequence=dates,
                              padding_value=self._max_sequence_length,
                              max_sequence_length=self._max_sequence_length)
        ages = _pad_sequence(sequence=ages,
                             padding_value=self._max_sequence_length,
                             max_sequence_length=self._max_sequence_length)
        visit_concept_orders = _pad_sequence(sequence=visit_concept_orders,
                                             padding_value=self._max_sequence_length - 1,
                                             max_sequence_length=self._max_sequence_length)

        output_mask = (token_ids == self._concept_tokenizer.get_unused_token_id()).astype(int)
        combined_label = np.stack([token_ids, output_mask], axis=-1)
        input_dict = {LayerInputNames.MASKED_CONCEPT_IDS: masked_token_ids,
                      LayerInputNames.CONCEPT_IDS: token_ids,
                      LayerInputNames.MASK: output_mask,
                      LayerInputNames.TIME_STAMPS: dates,
                      LayerInputNames.AGES: ages,
                      LayerInputNames.VISIT_SEGMENTS: visit_segments,
                      LayerInputNames.VISIT_CONCEPT_ORDERS: visit_concept_orders}

        output_dict = {LayerInputNames.CONCEPT_PREDICTIONS: combined_label}
        return input_dict, output_dict

    def _mask_concepts(self, concepts):
        masked_concepts = concepts.copy()
        output_mask = np.zeros((self._max_sequence_length,), dtype=int)
        if self._is_training:
            for word_pos in range(0, len(concepts)):
                if concepts[word_pos] == self._concept_tokenizer.get_unused_token_id():
                    break

                if random.random() < 0.15:
                    dice = random.random()
                    if dice < 0.8:
                        masked_concepts[word_pos] = self._concept_tokenizer.get_mask_token_id()
                    elif dice < 0.9:
                        masked_concepts[word_pos] = random.randint(
                            self._concept_tokenizer.get_first_token_id(),
                            self._concept_tokenizer.get_last_token_id())
                    # else: 10% of the time we just leave the token as is
                    output_mask[word_pos] = 1

        return masked_concepts, output_mask
