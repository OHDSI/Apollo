import json

from numpy.typing import NDArray
from torch.utils.data import Dataset
from typing import List

import numpy as np

PADDING_TOKEN = "[PAD]"
MASK_TOKEN = "[MASK]"
OUT_OF_VOCABULARY_TOKEN = "[OOV]"
CLASSIFICATION_TOKEN = "[CLS]"


class ConceptTokenizer:
    """
    Maps concept ID strings and special tokens to integer indexes and vice versa.
    """

    def __init__(self):
        self._word_index = {}
        self._index_word = {}
        self._padding_token_index = 0
        self._mask_token_index = 1
        self._oov_token_index = 2
        self._classification_token_index = 3

    def fit_on_concept_sequences(self, dataset: Dataset, column_name: str):
        """
        Fit the tokenizer on the concept IDs in the given column of the given parquet data iterator.
        Args:
            dataset: The dataset to fit on.
            column_name: The name of the column containing the concept IDs.
        """
        words = set()
        for row in dataset:
            for concept_id in row[column_name]:
                words.add(concept_id)
        vocabulary = [PADDING_TOKEN, MASK_TOKEN, OUT_OF_VOCABULARY_TOKEN, CLASSIFICATION_TOKEN]
        vocabulary.extend(words)
        self._word_index = dict(zip(vocabulary, list(range(0, len(vocabulary)))))
        self._index_word = {index: word for word, index in self._word_index.items()}
        self._oov_token_index = self._word_index[OUT_OF_VOCABULARY_TOKEN]
        self._padding_token_index = self._word_index[PADDING_TOKEN]
        self._mask_token_index = self._word_index[MASK_TOKEN]

    def encode(self, concept_ids: NDArray[str]) -> NDArray[np.int64]:
        result = np.empty(len(concept_ids), dtype=np.int64)
        for i in range(len(concept_ids)):
            idx = self._word_index.get(concept_ids[i])
            if idx is None:
                result[i] = self._oov_token_index
            else:
                result[i] = idx
        return result

    def decode(self, concept_token_ids: List[int]) -> List[str]:
        return [self._index_word.get(i) for i in concept_token_ids]

    def get_vocab_size(self):
        return len(self._word_index)

    def get_padding_token_id(self):
        return self._padding_token_index

    def get_mask_token_id(self) -> int:
        return self._mask_token_index

    def get_out_of_vocabulary_token_id(self) -> int:
        return self._oov_token_index

    def get_classification_token_id(self) -> int:
        return self._classification_token_index

    def get_first_token_id(self) -> int:
        return 4

    def get_last_token_id(self) -> int:
        return self.get_vocab_size() - 1

    def save_to_json(self, file_name: str) -> None:
        with open(file_name, "w") as f:
            json.dump(self._word_index, f)


def load_from_json(file_name: str) -> ConceptTokenizer:
    self = ConceptTokenizer()
    with open(file_name, "r") as f:
        self._word_index = json.load(f)
    self._index_word = {index: word for word, index in self._word_index.items()}
    self._oov_token_index = self._word_index[OUT_OF_VOCABULARY_TOKEN]
    self._padding_token_index = self._word_index[PADDING_TOKEN]
    self._mask_token_index = self._word_index[MASK_TOKEN]
    return self
