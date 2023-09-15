import shutil
import tempfile
import os

import numpy as np
import unittest
import pyarrow as pa
from torch.utils.data import DataLoader

from data_loading import tokenizer, learning_objectives, parquet_data_iterator, dataset, data_transformer


class TestDataGenerating(unittest.TestCase):

    def setUp(self) -> None:
        self.parquet_folder = tempfile.mkdtemp()

        row = pa.Table.from_pydict(
            {
                "cohort_member_id": [1],
                "person_id": [1],
                "concept_ids": [["VS", "123", "456", "VE", "W1", "VS", "456", "VE"]],
                "visit_segments": [np.array([2, 2, 2, 2, 0, 1, 1, 1], dtype=np.int32)],
                "orders": [np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int32)],
                "dates": [np.array([1800, 1800, 1800, 1800, 0, 1801, 1801, 1801], dtype=np.int32)],
                "ages": [np.array([75, 75, 75, 75, 0, 76, 76, 76], dtype=np.int32)],
                "visit_concept_orders": [np.array([1, 1, 1, 1, 2, 2, 2, 2], dtype=np.int32)],
                "num_of_concepts": [8],
                "num_of_visits": [2],
                "visit_concept_ids": [np.array(["9202", "9202", "9202", "9202", "0", "9202", "9202", "9202"],
                                               dtype=str)],
            }
        )
        pa.parquet.write_table(row, os.path.join(self.parquet_folder, "test.parquet"))

    def tearDown(self):
        shutil.rmtree(self.parquet_folder)

    def test_parquest_data_iterator(self):
        data = parquet_data_iterator.ParquetDataIterator([os.path.join(self.parquet_folder, "test.parquet")], None)
        row_count = 0
        for row in data:
            assert row["cohort_member_id"] == 1
            assert row["person_id"] == 1
            assert len(row["concept_ids"]) == 8
            assert len(row["visit_segments"]) == 8
            assert len(row["orders"]) == 8
            assert len(row["dates"]) == 8
            assert len(row["ages"]) == 8
            assert len(row["visit_concept_orders"]) == 8
            assert row["num_of_concepts"] == 8
            assert row["num_of_visits"] == 2
            assert len(row["visit_concept_ids"]) == 8
            row_count += 1
            break
        assert row_count == 1

    def test_concept_tokenizer(self):
        ds = dataset.ApolloDataset(folder=self.parquet_folder, train_test_split=1, is_train=True)
        concept_tokenizer = tokenizer.ConceptTokenizer()
        concept_tokenizer.fit_on_concept_sequences(ds, "concept_ids")
        assert concept_tokenizer.encode(np.array([tokenizer.OUT_OF_VOCABULARY_TOKEN])) == \
               [concept_tokenizer.get_out_of_vocabulary_token_id()]
        assert concept_tokenizer.encode(np.array([tokenizer.MASK_TOKEN])) == \
               [concept_tokenizer.get_mask_token_id()]
        assert concept_tokenizer.encode(np.array([tokenizer.PADDING_TOKEN])) == \
               [concept_tokenizer.get_padding_token_id()]
        test_concept_ids = np.array(["VS", "123", "456", "VE", "W1", "VS", "456", "VE"], dtype=str)
        encoding = concept_tokenizer.encode(test_concept_ids)
        decoding = concept_tokenizer.decode(encoding.tolist())
        assert decoding == test_concept_ids.tolist()

        json_filename = os.path.join(self.parquet_folder, "_test.json")
        concept_tokenizer.save_to_json(json_filename)
        concept_tokenizer_2 = tokenizer.load_from_json(json_filename)
        os.remove(json_filename)
        encoding_2 = concept_tokenizer_2.encode(test_concept_ids)
        assert (encoding == encoding_2).all()

    def test_data_loader(self):
        ds = dataset.ApolloDataset(folder=self.parquet_folder, train_test_split=1, is_train=True)
        concept_tokenizer = tokenizer.ConceptTokenizer()
        concept_tokenizer.fit_on_concept_sequences(ds, "concept_ids")
        visit_concept_tokenizer = tokenizer.ConceptTokenizer()
        visit_concept_tokenizer.fit_on_concept_sequences(ds, "visit_concept_ids")
        learning_objectives_ = [learning_objectives.MaskedConceptLearningObjective(concept_tokenizer),
                                learning_objectives.MaskedVisitConceptLearningObjective(visit_concept_tokenizer)]
        dt = data_transformer.ApolloDataTransformer(learning_objectives_)
        ds = dataset.ApolloDataset(folder=self.parquet_folder,
                                   data_transformer=dt,
                                   train_test_split=1,
                                   is_train=True)
        data_loader = DataLoader(ds, batch_size=1)
        batch_count = 0
        for inputs, outputs in data_loader:
            # TODO: add some specific tests on output
            batch_count += 1
            break
        assert batch_count == 1
