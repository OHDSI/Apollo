import json

from data_generating.parquet_data_iterator import ParquetDataIterator
from typing import List

UNUSED_TOKEN = "[UNUSED]"
MASK_TOKEN = "[MASK]"
OUT_OF_VOCABULARY_TOKEN = "[OOV]"


class ConceptTokenizer:
    """
    Maps concept ID strings and special tokens to integer indexes and vice versa.
    """

    def __init__(self):
        self._word_index = {}
        self._index_word = {}
        self._oov_token_index = 1
        self._unused_token_index = 2
        self._mask_token_index = 3

    def fit_on_concept_sequences(self, parquet_data_iterator: ParquetDataIterator, column_name: str):
        """
        Fit the tokenizer on the concept IDs in the given column of the given parquet data iterator.
        Args:
            parquet_data_iterator: A parquet data iterator.
            column_name: The name of the column containing the concept IDs.
        """
        words = set()
        for row in parquet_data_iterator:
            for concept_id in row[column_name]:
                words.add(concept_id)
        vocabulary = [OUT_OF_VOCABULARY_TOKEN, UNUSED_TOKEN, MASK_TOKEN]
        vocabulary.extend(words)
        self._word_index = dict(zip(vocabulary, list(range(1, len(vocabulary) + 1))))
        self._index_word = {index: word for word, index in self._word_index.items()}
        self._oov_token_index = self._word_index[OUT_OF_VOCABULARY_TOKEN]
        self._unused_token_index = self._word_index[UNUSED_TOKEN]
        self._mask_token_index = self._word_index[MASK_TOKEN]

    def encode(self, concept_ids: List[str]) -> List[int]:
        result = []
        for w in concept_ids:
            i = self._word_index.get(w)
            if i is None:
                result.append(self._oov_token_index)
            else:
                result.append(i)
        return result

    def decode(self, concept_token_ids: List[int]) -> List[str]:
        return [self._index_word.get(i) for i in concept_token_ids]

    def get_vocab_size(self):
        return len(self._word_index)

    def get_unused_token_id(self):
        return self._unused_token_index

    def get_mask_token_id(self):
        return self._mask_token_index

    def get_out_of_vocabulary_token_id(self):
        return self._oov_token_index

    def save_to_json(self, file_name: str):
        with open(file_name, "w") as f:
            json.dump(self._word_index, f)


def load_from_json(file_name: str):
    self = ConceptTokenizer()
    with open(file_name, "r") as f:
        self._word_index = json.load(f)
    self._index_word = {index: word for word, index in self._word_index.items()}
    self._oov_token_index = self._word_index[OUT_OF_VOCABULARY_TOKEN]
    self._unused_token_index = self._word_index[UNUSED_TOKEN]
    self._mask_token_index = self._word_index[MASK_TOKEN]
    return self
