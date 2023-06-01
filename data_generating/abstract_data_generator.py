from abc import ABC, abstractmethod

from data_generating.parquet_data_iterator import ParquetDataIterator


class AbstractDataGenerator(ABC):
    """
    The interface of the data generator that will be visible to the learning objectives.
    """
    @abstractmethod
    def get_parquet_data_iterator(self) -> ParquetDataIterator:
        pass

    @abstractmethod
    def get_is_training(self) -> bool:
        pass

    @abstractmethod
    def get_max_sequence_length(self) -> int:
        pass
