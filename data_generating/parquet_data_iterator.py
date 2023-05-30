from typing import Iterator

import pandas as pd
import pyarrow.parquet as pq


class ParquetDataIterator:
    """Iterate over rows in parquet files"""

    def __init__(self, parquet_folder_name: str):
        """
        Initialization

        Args:
            parquet_folder_name: Path to the folder containing the parquet files.
        """
        self._parquet_path = parquet_folder_name
        self._dataset = pq.ParquetDataset(parquet_folder_name)
        self._nrows = sum(fragment.count_rows() for fragment in self._dataset.fragments)
        self._iterator = self.__iter__()  # Used by __next__

    def __iter__(self) -> Iterator[pd.DataFrame]:
        for fragment in self._dataset.fragments:
            for batch in fragment.to_batches():
                for index, row in batch.to_pandas().iterrows():
                    yield row

    def __len__(self) -> int:
        """The number of batches per epoch"""
        return self._nrows

    def __next__(self) -> pd.DataFrame:
        return next(self._iterator)
