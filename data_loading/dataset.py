import math
from typing import List, Optional, Iterable, Dict
import os

import torch.utils.data as data

from data_loading.parquet_data_iterator import ParquetDataIterator
from data_loading.data_transformer import ApolloDataTransformer


def list_files_with_extension(folder_path:  str, extension: str) -> List[str]:
    file_list = []
    with os.scandir(folder_path) as entries:
        for entry in entries:
            if entry.is_file() and entry.name.endswith(extension):
                file_list.append(os.path.join(folder_path, entry.name))
    return file_list


class ApolloDataset(data.IterableDataset):
    """
    A dataset for PyTorch that iterates over rows in parquet files. The Parquet files are split into training and test
    data. If the dataset is used across multiple workers, each worker will get a different subset of the data.
    """

    def __init__(self,
                 folder: str,
                 data_transformer: Optional[ApolloDataTransformer] = None,
                 train_test_split: float = 0.8,
                 is_train: bool = True):
        """
        Initialization

        Args:
            folder: Path to the folder containing the parquet files.
            data_transformer: A data transformer that will be used to transform the raw data into the format required
            for the various learning objectives.
            train_test_split: The fraction of the data that should be used for training. The rest is used for testing.
            is_train: If true, this class will return the training data, otherwise the test data.
        """
        if not os.path.exists(folder):
            raise ValueError(f"Folder '{folder}' does not exist")
        self._folder = folder
        self._data_transformer = data_transformer
        self._train_test_split = train_test_split
        self._is_train = is_train
        all_files = list_files_with_extension(folder, ".parquet")
        split_point = int(len(all_files) * train_test_split)
        if is_train:
            self._files = all_files[:split_point]
        else:
            self._files = all_files[split_point:]

    def __iter__(self) -> Iterable[Dict]:
        worker_info = data.get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
            return ParquetDataIterator(self._files, self._data_transformer)
        else:  # in a worker process
            per_worker = int(math.ceil(len(self._files) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, len(self._files))
            return ParquetDataIterator(self._files[iter_start:iter_end], self._data_transformer)


if __name__ == "__main__":
    trainDataset = ApolloDataset("d:/GPM_Sim/pretraining/patient_sequence", is_train=True)
    testDataset = ApolloDataset("d:/GPM_Sim/pretraining/patient_sequence", is_train=False)
    for row in trainDataset:
        print(row)
        break
