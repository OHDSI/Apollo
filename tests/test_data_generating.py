import shutil
import tempfile
import os

import pandas as pd
import numpy as np
import unittest
import tensorflow as tf

from data_loading import tokenizer, learning_objective, data_generator, parquet_data_iterator


class TestDataGenerating(unittest.TestCase):

    def setUp(self) -> None:
        self.parquet_folder = tempfile.mkdtemp()

        row = pd.DataFrame(
            {
                "cohort_member_id": [1],
                "person_id": [1],
                "concept_ids": [np.array(["VS", "123", "456", "VE", "W1", "VS", "456", "VE"], dtype=str)],
                "visit_segments": [np.array([2, 2, 2, 2, 0, 1, 1, 1], dtype=np.int32)],
                "orders": [np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int32)],
                "dates": [np.array([1800, 1800, 1800, 1800, 0, 1801, 1801, 1801], dtype=np.int32)],
                "ages": [np.array([75, 75, 75, 75, 0, 76, 76, 76], dtype=np.int32)],
                "visit_concept_orders": [np.array([1, 1, 1, 1, 2, 2, 2, 2], dtype=np.int32)],
                "num_of_concepts": [8],
                "num_of_visits": [2],
                "visit_concept_ids": [np.array(["9202", "9202", "9202", "9202", "0", "9202", "9202", "9202"], dtype=str)],
            }
        )
        row.to_parquet(os.path.join(self.parquet_folder, "test.parquet"))

    def tearDown(self):
        shutil.rmtree(self.parquet_folder)

    def test_parquest_data_iterator(self):
        data = parquet_data_iterator.ParquetDataIterator(parquet_folder_name=self.parquet_folder)
        row_count = 0
        for row in data:
            assert row["cohort_member_id"] == 1
            assert row["person_id"] == 1
            assert row["concept_ids"].shape == (8,)
            assert row["visit_segments"].shape == (8,)
            assert row["orders"].shape == (8,)
            assert row["dates"].shape == (8,)
            assert row["ages"].shape == (8,)
            assert row["visit_concept_orders"].shape == (8,)
            assert row["num_of_concepts"] == 8
            assert row["num_of_visits"] == 2
            assert row["visit_concept_ids"].shape == (8,)
            row_count += 1
            break
        assert row_count == 1


    def test_concept_tokenizer(self):
        data = parquet_data_iterator.ParquetDataIterator(parquet_folder_name=self.parquet_folder)
        concept_tokenizer = tokenizer.ConceptTokenizer()
        concept_tokenizer.fit_on_concept_sequences(parquet_data_iterator=data,
                                                   column_name="concept_ids")
        assert concept_tokenizer.encode([tokenizer.OUT_OF_VOCABULARY_TOKEN]) == \
               [concept_tokenizer.get_out_of_vocabulary_token_id()]
        assert concept_tokenizer.encode([tokenizer.MASK_TOKEN]) == [concept_tokenizer.get_mask_token_id()]
        assert concept_tokenizer.encode([tokenizer.PADDING_TOKEN]) == [concept_tokenizer.get_padding_token_id()]
        test_concept_ids = np.array(["VS", "123", "456", "VE", "W1", "VS", "456", "VE"], dtype=str)
        encoding = concept_tokenizer.encode(test_concept_ids)
        decoding = concept_tokenizer.decode(encoding)
        assert decoding == test_concept_ids.tolist()

        json_filename = os.path.join(self.parquet_folder, "_test.json")
        concept_tokenizer.save_to_json(json_filename)
        concept_tokenizer_2 = tokenizer.load_from_json(json_filename)
        os.remove(json_filename)
        encoding_2 = concept_tokenizer_2.encode(test_concept_ids)
        assert encoding == encoding_2


    def test_data_generator(self):
        learning_objectives = [learning_objective.MaskedLanguageModelLearningObjective(work_folder=self.parquet_folder),
                               learning_objective.VisitPredictionLearningObjective(work_folder=self.parquet_folder)]
        bert_data_generator = data_generator.DataGenerator(training_data_path=self.parquet_folder,
                                                           batch_size=4,
                                                           max_sequence_length=10,
                                                           min_sequence_length=5,
                                                           is_training=False,
                                                           learning_objectives=learning_objectives)
        assert len(bert_data_generator) == 1
        batch_count = 0
        for batch in bert_data_generator.generator():
            # TODO: add some specific tests on output
            batch_count += 1
            break
        assert batch_count == 1
