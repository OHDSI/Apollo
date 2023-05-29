import tempfile
import os

import pandas as pd
import numpy as np
import pytest

from data_generating import tokenizer
from data_generating.parquet_data_iterator import ParquetDataIterator


@pytest.fixture(scope="session")
def parquet_folder() -> tempfile.TemporaryDirectory:
    """
    Create a parquet file for testing.
    Returns: A temporary directory object containing a parquet file. The directory is deleted when the test is done.
    """
    temp_folder = tempfile.TemporaryDirectory()
    row = pd.Series(
        {
            "cohort_member_id": 1,
            "person_id": 1,
            "concept_ids": np.array(["VS", "123", "456", "VE", "W1", "VS", "456", "VE"], dtype=str),
            "visit_segments": np.array([2, 2, 2, 2, 0, 1, 1, 1], dtype=np.int32),
            "orders": np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int32),
            "dates": np.array([1800, 1800, 1800, 1800, 0, 1801, 1801, 1801], dtype=np.int32),
            "ages": np.array([75, 75, 75, 75, 0, 76, 76, 76], dtype=np.int32),
            "visit_concept_orders": np.array([1, 1, 1, 1, 2, 2, 2, 2], dtype=np.int32),
            "num_of_concepts": 8,
            "num_of_visits": 2,
            "visit_concept_ids": np.array(["9202", "9202", "9202", "9202", "0", "9202", "9202", "9202"], dtype=str),
        }
    ).to_frame().transpose()
    row.to_parquet(os.path.join(temp_folder.name + "test.parquet"))
    return temp_folder


def test_parquest_data_iterator(parquet_folder: tempfile.TemporaryDirectory):
    data = ParquetDataIterator(parquet_folder_name=parquet_folder.name)
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
        break


def test_concept_tokenizer(parquet_folder: tempfile.TemporaryDirectory):
    data = ParquetDataIterator(parquet_folder_name=parquet_folder.name)
    concept_tokenizer = tokenizer.ConceptTokenizer()
    concept_tokenizer.fit_on_concept_sequences(parquet_data_iterator=data,
                                               column_name="concept_ids")
    assert concept_tokenizer.encode([tokenizer.OUT_OF_VOCABULARY_TOKEN]) == [concept_tokenizer.get_out_of_vocabulary_token_id()]
    assert concept_tokenizer.encode([tokenizer.MASK_TOKEN]) == [concept_tokenizer.get_mask_token_id()]
    assert concept_tokenizer.encode([tokenizer.UNUSED_TOKEN]) == [concept_tokenizer.get_unused_token_id()]
    test_concept_ids = np.array(["VS", "123", "456", "VE", "W1", "VS", "456", "VE"], dtype=str)
    encoding = concept_tokenizer.encode(test_concept_ids)
    decoding = concept_tokenizer.decode(encoding)
    assert decoding == test_concept_ids.tolist()

    json_file = os.path.join(parquet_folder.name, "concept_tokenizer.json")
    concept_tokenizer.save_to_json(json_file)
    concept_tokenizer_2 = tokenizer.load_from_json(json_file)
    encoding_2 = concept_tokenizer_2.encode(test_concept_ids)
    assert encoding == encoding_2


def test_data_generator(parquet_folder: tempfile.TemporaryDirectory):
    # TODO: add more tests
    pass
