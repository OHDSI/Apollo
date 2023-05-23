import sys
from typing import Dict, List
import math
import datetime as dt
import cProfile
import logging
import configparser

import pandas as pd

from cdm_processing.abstract_cdm_processor import AbstractToParquetCdmDataProcessor
import cdm_processing.cdm_processor_utils as cpu

PERSON = "person"
START_DATE = "start_date"
CONCEPT_ID = "concept_id"
DRUG_EXPOSURE = "drug_exposure"
DRUG_CONCEPT_ID = "drug_concept_id"
VISIT_START = "VS"
VISIT_END = "VE"
EPOCH = dt.date(1970, 1, 1)


class ProcessingStatistics:
    """
    A class for storing and reporting out some statistics about the CDM processing.
    """

    def __init__(self):
        self.mapped_by_id = 0
        self.mapped_by_date = 0
        self.mapped_to_new_visit = 0
        self.existing_visits = 0
        self.new_visits = 0
        self.removed_concept_rows = 0
        self.persons = 0
        self.observation_periods = 0

    def record_visit_mapping_stats(self, visit_data: cpu.VisitData):
        self.mapped_by_id += visit_data.mapped_by_id
        self.mapped_by_date += visit_data.mapped_by_date
        self.mapped_to_new_visit += visit_data.mapped_to_new_visit
        if visit_data.new_visit:
            self.new_visits += 1
        else:
            self.existing_visits += 1

    def record_removed_concept_rows(self, row_count: int):
        self.removed_concept_rows += row_count

    def record_person(self):
        self.persons += 1

    def record_observation_period(self):
        self.observation_periods += 1

    def log_statistics(self, partition_i: int):
        logging.debug("Partition %s persons: %s", partition_i, self.persons)
        logging.debug("Partition %s observation periods: %s", partition_i, self.observation_periods)
        logging.debug("Partition %s events mapped to visit by ID: %s", partition_i, self.mapped_by_id)
        logging.debug("Partition %s events mapped to visit by date: %s", partition_i, self.mapped_by_date)
        logging.debug("Partition %s events mapped to new visits: %s", partition_i, self.mapped_to_new_visit)
        logging.debug("Partition %s existing visits: %s", partition_i, self.existing_visits)
        logging.debug("Partition %s newly created visits: %s", partition_i, self.new_visits)
        logging.debug("Partition %s removed events having unwanted concept ID: %s", partition_i, self.removed_concept_rows)


class OutputRow:
    """
    A class representing the input format expected for the CEHR-BERT trainer.
    """

    def __init__(self):
        self.cohort_member_id = 0
        self.person_id = 0
        self.concept_ids = []
        self.visit_segments = []
        self.dates = []
        self.ages = []
        self.visit_concept_orders = []
        self.visit_concept_ids = []
        self.orders = []
        self.num_of_visits = 0
        self.num_of_concepts = 0

    def to_pandas(self) -> pd.DataFrame:
        output_row = pd.Series(
            {
                "cohort_member_id": self.cohort_member_id,
                "person_id": self.person_id,
                "concept_ids": self.concept_ids,
                "visit_segments": self.visit_segments,
                "orders": self.orders,
                "dates": self.dates,
                "ages": self.ages,
                "visit_concept_orders": self.visit_concept_orders,
                "num_of_visits": self.num_of_visits,
                "num_of_concepts": self.num_of_concepts,
                "visit_concept_ids": self.visit_concept_ids,
            }
        ).to_frame().transpose()
        return output_row


def _create_interval_token(days: int) -> str:
    if days < 0:
        return "W-1"
    if days < 28:
        return f"W{str(math.floor(days / 7))}"
    if days < 360:
        return f"M{str(math.floor(days / 30))}"
    return "LT"


def _days_to_weeks(days: int) -> int:
    return math.floor(max(days, 0) / 7)


def _days_to_months(days: int) -> int:
    return math.floor(days / 30.5)


class CehrBertCdmDataProcessor(AbstractToParquetCdmDataProcessor):
    """
    A re-implementation of the processor for CEHR-BERT (https://github.com/cumc-dbmi/cehr-bert)
    """

    def __init__(self, cdm_data_path: str, output_path: str, max_cores: int = -1,
                 map_drugs_to_ingredients: bool = False, concepts_to_remove: List[int] = [0]):
        super(AbstractToParquetCdmDataProcessor, self).__init__(
            cdm_data_path=cdm_data_path, output_path=output_path, max_cores=max_cores
        )
        self._map_drugs_to_ingredients = map_drugs_to_ingredients
        self._concepts_to_remove = concepts_to_remove

    def _prepare(self):
        super()._prepare()
        if self._map_drugs_to_ingredients:
            self._drug_mapping = cpu.load_mapping_to_ingredients(self._cdm_data_path)

    def _prepare_partition(self, partition_i: int):
        super()._prepare_partition(partition_i=partition_i)
        self._processing_statistics = ProcessingStatistics()

    def _finish_partition(self, partition_i: int):
        self._processing_statistics.log_statistics(partition_i=partition_i)
        super()._finish_partition(partition_i=partition_i)

    def _process_person(self, person_id: int, cdm_tables: Dict[str, pd.DataFrame]):
        self._processing_statistics.record_person()
        cdm_tables, removed_row_counts = cpu.remove_concepts(cdm_tables=cdm_tables,
                                                             concept_ids=self._concepts_to_remove)
        self._processing_statistics.record_removed_concept_rows(sum(removed_row_counts.values()))
        cpu.call_per_observation_period(
            cdm_tables=cdm_tables, function=self._process_observation_period
        )

    def _process_observation_period(
            self, observation_period: pd.Series, cdm_tables: Dict[str, pd.DataFrame]
    ):
        self._processing_statistics.record_observation_period()
        if self._map_drugs_to_ingredients and DRUG_EXPOSURE in cdm_tables:
            cdm_tables[DRUG_EXPOSURE] = cpu.map_concepts(cdm_table=cdm_tables[DRUG_EXPOSURE],
                                                         concept_id_field=DRUG_CONCEPT_ID,
                                                         mapping=self._drug_mapping)
        date_of_birth = cpu.get_date_of_birth(person=cdm_tables[PERSON].iloc[0])
        output_row = OutputRow()
        output_row.cohort_member_id = observation_period[cpu.OBSERVATION_PERIOD_ID]
        output_row.person_id = observation_period[cpu.PERSON_ID]
        # Init with random date to silence code warning:
        previous_visit_end_date = dt.date(2000, 1, 1)
        visit_rank = 0
        for visit_group in cpu.group_by_visit(
                cdm_tables=cdm_tables,
                link_by_date=True,
                create_missing_visits=True,
                missing_visit_concept_id=0,
        ):
            visit_rank += 1
            if visit_rank > 1:
                # Add interval token:
                interval_token = _create_interval_token((visit_group.visit_start_date - previous_visit_end_date).days)
                output_row.concept_ids.append(interval_token)
                output_row.visit_segments.append(0)
                output_row.dates.append(0)
                output_row.ages.append(-1)
                output_row.visit_concept_orders.append(visit_rank + 1)
                output_row.visit_concept_ids.append(0)
            visit_end_date = visit_group.visit["visit_end_date"]
            event_table = cpu.union_domain_tables(visit_group.cdm_tables)
            visit_token_len = len(event_table) + 2
            output_row.concept_ids.append(VISIT_START)
            output_row.concept_ids.extend(event_table[CONCEPT_ID].astype(str).to_list())
            output_row.concept_ids.append(VISIT_END)
            output_row.visit_segments.extend([visit_rank % 2 + 1] * visit_token_len)
            output_row.dates.append(_days_to_weeks((visit_group.visit_start_date - EPOCH).days))
            output_row.dates.extend(event_table[START_DATE].apply(lambda x: _days_to_weeks((x - EPOCH).days)))
            output_row.dates.append(_days_to_weeks((visit_end_date - EPOCH).days))
            output_row.ages.append(_days_to_months((visit_group.visit_start_date - date_of_birth).days))
            output_row.ages.extend(event_table[START_DATE].apply(lambda x: _days_to_months((x - date_of_birth).days)))
            output_row.ages.append(_days_to_months((visit_end_date - date_of_birth).days))
            output_row.visit_concept_orders.extend([visit_rank] * visit_token_len)
            output_row.visit_concept_ids.extend([visit_group.visit[cpu.VISIT_CONCEPT_ID]] * visit_token_len)
            previous_visit_end_date = visit_end_date
            self._processing_statistics.record_visit_mapping_stats(visit_group)
        output_row.orders = list(range(0, len(output_row.concept_ids)))
        output_row.num_of_visits = visit_rank
        output_row.num_of_concepts = len(output_row.concept_ids)
        if (len(output_row.concept_ids) > 0):
            self._output.append(output_row.to_pandas())


def main(args):
    config = configparser.ConfigParser()
    config.read(args[0])
    my_cdm_data_processor = CehrBertCdmDataProcessor(
        cdm_data_path=config["system"].get("cdm_data_path"),
        max_cores=config["system"].getint("max_cores"),
        output_path=config["system"].get("output_path"),
        map_drugs_to_ingredients=config["mapping"].getboolean("map_drugs_to_ingredients"),
        concepts_to_remove=[int(x) for x in config["mapping"].get("concepts_to_remove").split(",")],
    )
    if config["debug"].getboolean("profile"):
        my_cdm_data_processor._max_cores = -1
        cProfile.run("my_cdm_data_processor.process_cdm_data()", "../stats")
    else:
        my_cdm_data_processor.process_cdm_data()


if __name__ == "__main__":
    main(sys.argv[1:])
