from typing import List, Iterator, Dict, Optional

import pyarrow.parquet as pq

from data_loading.data_transformer import ApolloDataTransformer


class ParquetDataIterator:
    """Iterate over rows in parquet files"""

    def __init__(self, parquet_files: List[str], data_transformer: Optional[ApolloDataTransformer]):
        """
        Initialization

        parquet_files: A list of full file names of Parquet files to iterate over.
        data_transformer: A data transformer that will be used to transform the raw data into the format required
        for the various learning objectives.

        """
        self._dataset = pq.ParquetDataset(parquet_files)
        self._data_transformer = data_transformer
        self._nrows = sum(fragment.count_rows() for fragment in self._dataset.fragments)
        self._iterator = self.__iter__()  # Used by __next__

    def __iter__(self) -> Iterator[Dict]:
        for fragment in self._dataset.fragments:
            for batch in fragment.to_batches():
                for row in batch.to_pylist():
                    if self._data_transformer is not None:
                        yield self._data_transformer.transform(row)
                    else:
                        yield row

    def __len__(self) -> int:
        """The number of batches per epoch"""
        return self._nrows

    def __next__(self) -> Dict:
        return next(self._iterator)
