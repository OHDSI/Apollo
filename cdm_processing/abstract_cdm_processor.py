import cProfile
from abc import ABC, abstractmethod
import multiprocessing
import os
from typing import Dict, Optional
import logging

import pyarrow.parquet as pq
import pyarrow as pa
import tqdm

from utils.logger import create_logger

PERSON = "person"
CDM_TABLES = [
    "person",
    "observation_period",
    "visit_occurrence",
    "condition_occurrence",
    "drug_exposure",
    "procedure_occurrence",
    "device_exposure",
    "measurement",
    "observation",
    "death",
]
LOGGER_FILE_NAME = "_cdm_processing_log.txt"  # Start with underscore so ignored by Parquet


class AbstractCdmDataProcessor(ABC):
    """
    An abstract class that implements iterating over partitioned data as generated by the
    GeneralPretrainedModelTools R package. It divides the partitions over various threads,
    and calls the _process_parition_cdm_data() function with all data of a partition.
    """

    def __init__(self,
                 cdm_data_path: str,
                 output_path: str,
                 max_cores: int = 1,
                 has_labels: bool = False,
                 label_subfolder: str = 'label'):
        """
        Args:
            cdm_data_path: The path where the CDM Parquet files are saved (using the GeneralPretrainModelTools packages).
            max_cores: The maximum number of CPU cores to use. If set to -1, all multihreading code will be bypassed for
               easier debugging.
            output_path: The path where the output files should be saved.
            has_labels: If true, the labels will be loaded from the label_subfolder.
            label_subfolder: The subfolder of the cdm_data_path where the labels are saved.
        """
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        self._cdm_data_path = cdm_data_path
        self._max_cores = max_cores
        self._person_partition_count = 0
        self._output_path = output_path
        self._profile = False
        self._has_labels = has_labels
        self._label_subfolder = label_subfolder
        self._configure_logger(clear_log_file=True)

    def set_profile(self, profile: bool):
        self._profile = profile

    def get_profile(self):
        return self._profile

    def _configure_logger(self, clear_log_file: bool = False):
        create_logger(log_file_name=os.path.join(self._output_path, LOGGER_FILE_NAME),
                      clear_log_file=clear_log_file)

    @abstractmethod
    def _process_partition_cdm_data(self,
                                   cdm_tables: Dict[str, pa.Table],
                                   labels: Optional[pa.Table],
                                   partition_i: int):
        # This functon is called for every parition (It is executed within a thread.)
        pass

    def process_cdm_data(self):
        """
        Process the CDM data in the provided cdm_data_path.
        """
        if self._profile:
            cProfile.runctx(statement="self._process_cdm_data()",
                            locals={"self": self},
                            globals={},
                            filename="../stats")
        else:
            self._process_cdm_data()

    def _process_cdm_data(self):
        self._get_partition_counts()
        if self._profile:
            logging.info("Profiling mode enabled, running first partition in single thread")
            self._process_partition(0)
        elif self._max_cores == 1:
            # Run single thread in main thread for easier debugging:
            for partition_i in range(self._person_partition_count):
                self._process_partition(partition_i)
        else:
            pool = multiprocessing.get_context("spawn").Pool(processes=self._max_cores)
            tasks = range(self._person_partition_count)
            work = self._process_partition
            for _ in tqdm.tqdm(pool.imap_unordered(work, tasks), total=len(tasks)):
                pass
            pool.close()
        logging.info("Finished processing data")

    def _get_partition_counts(self):
        files = os.listdir(os.path.join(self._cdm_data_path, PERSON))
        self._person_partition_count = len(
            list(filter(lambda x: ".parquet" in x, files))
        )
        logging.info("Found %s partitions", self._person_partition_count)

    def _process_partition(self, partition_i: int):
        # This function is executed within a thread
        # Need to re-configure logger because we're in a thread:
        self._configure_logger()
        logging.debug("Starting partition %s of %s", partition_i, self._person_partition_count)

        file_name = "part{:04d}.parquet".format(partition_i + 1)
        available_tables = [table for table in CDM_TABLES if table in os.listdir(self._cdm_data_path)]
        cdm_tables = {table: pq.read_table(os.path.join(self._cdm_data_path, table, file_name)) for table in
                      available_tables}
        if self._has_labels:
            labels = pq.read_table(os.path.join(self._cdm_data_path, self._label_subfolder, file_name))
        else:
            labels = None
        self._process_partition_cdm_data(cdm_tables=cdm_tables, labels=labels, partition_i=partition_i)

        logging.debug("Finished partition %s of %s", partition_i, self._person_partition_count)
