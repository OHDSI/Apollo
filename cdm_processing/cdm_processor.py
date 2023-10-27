import os
import sys
from typing import Dict, List, Optional
import logging
import configparser

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from cdm_processing.abstract_cdm_processor import AbstractCdmDataProcessor
from cdm_processing.cdm_processing_settings import CdmProcessingSettings
import cdm_processing.cdm_processor_utils as cdm_utils
import utils.logger as logger


class CdmDataProcessor(AbstractCdmDataProcessor):
    """
    A re-implementation of the processor for CEHR-BERT (https://github.com/cumc-dbmi/cehr-bert)
    """

    def __init__(self, settings: CdmProcessingSettings):
        super().__init__(
            cdm_data_path=settings.cdm_data_path,
            output_path=settings.output_path,
            max_cores=settings.max_cores,
            has_labels=settings.has_labels,
            label_subfolder=settings.label_sub_folder
        )
        self._settings = settings
        if settings.map_drugs_to_ingredients:
            self._drug_mapping = cdm_utils.load_mapping_to_ingredients(self._cdm_data_path)
        if settings.profile:
            self.set_profile(True)
        logger.log_settings(settings)
        settings.write_mapping_settings(os.path.join(settings.output_path, "cdm_mapping.ini"))

    def _process_partition_cdm_data(self,
                                    cdm_tables: Dict[str, pa.Table],
                                    labels: Optional[pa.Table],
                                    partition_i: int):
        """
        Process a single partition of CDM data, and save the result to disk.
        """
        cdm_tables["person"] = cdm_utils.add_date_of_birth(cdm_tables["person"])
        if self._settings.map_drugs_to_ingredients:
            cdm_tables["drug_exposure"] = cdm_utils.map_concepts(cdm_table=cdm_tables["drug_exposure"],
                                                                 concept_id_field="drug_concept_id",
                                                                 mapping=self._drug_mapping)
        event_table = cdm_utils.union_domain_tables(cdm_tables)
        event_table, removed_concepts = cdm_utils.remove_concepts(event_table=event_table,
                                                                  concept_ids=self._settings.concepts_to_remove)
        event_table, removed_duplicates = cdm_utils.remove_duplicates(event_table=event_table)
        event_table, visit_occurrence, mapping_stats = cdm_utils.link_events_to_visits(event_table=event_table,
                                                                                       visit_occurrence=cdm_tables[
                                                                                           "visit_occurrence"],
                                                                                       mising_visit_concept_id=1)
        cdm_tables["visit_occurrence"] = visit_occurrence
        sequence_data = self._create_sequence_tables(cdm_tables=cdm_tables, event_table=event_table, labels=labels)
        file_name = "part{:04d}.parquet".format(partition_i + 1)
        pq.write_table(sequence_data, os.path.join(self._output_path, file_name))

        logging.debug("Partition %s persons: %s", partition_i, len(cdm_tables["person"]))
        logging.debug("Partition %s observation periods: %s", partition_i, len(cdm_tables["observation_period"]))
        logging.debug("Partition %s removed events having unwanted concept ID: %s", partition_i, removed_concepts)
        logging.debug("Partition %s removed duplicate events: %s", partition_i, removed_duplicates)
        logging.debug("Partition %s events mapped to visit by ID: %s", partition_i, mapping_stats["mapped_by_id"])
        logging.debug("Partition %s events mapped to visit by date: %s", partition_i, mapping_stats["mapped_by_date"])
        logging.debug("Partition %s events mapped to new visits: %s", partition_i, mapping_stats["mapped_to_new_visit"])
        logging.debug("Partition %s existing visits: %s", partition_i, mapping_stats["existing_visits"])
        logging.debug("Partition %s newly created visits: %s", partition_i, mapping_stats["new_visits"])

    def _create_sequence_tables(self,
                                cdm_tables: Dict[str, pa.Table],
                                event_table: pa.Table,
                                labels: Optional[pa.Table]
                                ) -> pa.Table:
        """
        Creates the table needed for the CEHR-BERT model.
        Args:
            cdm_tables: A dictionary of CDM tables, with the table name as key and the table as value. The person table
                should have the date_of_birth field added using the add_date_of_birth_to_person_table() function.
            event_table: The table with the clinical events. This is a combination across the CDM domain tables, as
                created by the union_cdm_tables() function.
            labels: The table with the labels.
        Returns:
            A table with the data needed for the CEHR-BERT model.
        """
        con = duckdb.connect(database=':memory:', read_only=False)
        con.execute("SET enable_progress_bar = false")
        con.register("visit_occurrence", cdm_tables["visit_occurrence"])
        con.register("observation_period_table", cdm_tables["observation_period"])
        con.register("person", cdm_tables["person"])
        con.register("event_table", event_table)
        if self._settings.has_labels:
            con.register("labels", labels)
        sql = "CREATE TABLE visits AS " \
              "SELECT visit_occurrence.*, " \
              "  observation_period_id, " \
              "  ROW_NUMBER() OVER (PARTITION BY observation_period_id ORDER BY visit_start_date) AS visit_rank " \
              "FROM visit_occurrence " \
              "INNER JOIN observation_period_table " \
              "  ON visit_occurrence.person_id = observation_period_table.person_id " \
              "    AND visit_occurrence.visit_start_date >= observation_period_table.observation_period_start_date " \
              "    AND visit_occurrence.visit_start_date <= observation_period_table.observation_period_end_date"
        con.execute(sql)
        sql = "CREATE TABLE interval_tokens AS " \
              "SELECT CASE " \
              "    WHEN days < 0 THEN 'W-1' " \
              "    WHEN days < 28 THEN 'W' || CAST(CAST(FLOOR(days / 7) AS INT) AS VARCHAR) " \
              "    WHEN days < 360 THEN 'M' || CAST(CAST(FLOOR(days / 30) AS INT) AS VARCHAR) " \
              "    ELSE 'LT' " \
              "  END AS concept_ids, " \
              "  0 AS visit_segments, " \
              "  0 AS dates, " \
              " -1 AS ages, " \
              "  visit_rank AS visit_concept_orders, " \
              "  CAST('0' AS VARCHAR) AS visit_concept_ids, " \
              "  -2 AS sort_order, " \
              "  observation_period_id, " \
              "  person_id " \
              "FROM (" \
              "  SELECT visits.visit_start_date - previous_visit.visit_end_date AS days," \
              "    visits.* " \
              "  FROM visits " \
              "  INNER JOIN visits previous_visit" \
              "    ON visits.observation_period_id = previous_visit.observation_period_id " \
              "      AND visits.visit_rank = previous_visit.visit_rank + 1" \
              ") intervals"
        con.execute(sql)
        sql = "CREATE TABLE start_tokens AS " \
              "SELECT 'VS' AS concept_ids, " \
              "  visit_rank % 2 + 1 AS visit_segments, " \
              "  DATE_DIFF('week', DATE '1970-01-01', visit_start_date) AS dates, " \
              "  DATE_DIFF('month', date_of_birth, visit_start_date) AS ages, " \
              "  visit_rank AS visit_concept_orders, " \
              "  CAST(visit_concept_id AS VARCHAR) AS visit_concept_ids, " \
              "  -1 AS sort_order, " \
              "  observation_period_id, " \
              "  person.person_id " \
              "FROM visits " \
              "INNER JOIN person " \
              "  ON visits.person_id = person.person_id"
        con.execute(sql)
        sql = "CREATE TABLE event_tokens AS " \
              "SELECT CAST(concept_id AS VARCHAR) AS concept_ids, " \
              "  visit_rank % 2 + 1 AS visit_segments, " \
              "  DATE_DIFF('week', DATE '1970-01-01', start_date) AS dates, " \
              "  DATE_DIFF('month', date_of_birth,start_date) AS ages, " \
              "  visit_rank AS visit_concept_orders, " \
              "  CAST(visit_concept_id AS VARCHAR) AS visit_concept_ids, " \
              "  concept_id AS sort_order, " \
              "  observation_period_id, " \
              "  person.person_id " \
              "FROM event_table " \
              "INNER JOIN visits " \
              "  ON event_table.internal_visit_id = visits.internal_visit_id " \
              "INNER JOIN person " \
              "  ON visits.person_id = person.person_id"
        con.execute(sql)
        sql = "CREATE TABLE end_tokens AS " \
              "SELECT 'VE' AS concept_ids, " \
              "  visit_rank % 2 + 1 AS visit_segments, " \
              "  DATE_DIFF('week', DATE '1970-01-01', visit_end_date) AS dates, " \
              "  DATE_DIFF('month', date_of_birth, visit_end_date) AS ages, " \
              "  visit_rank AS visit_concept_orders, " \
              "  CAST(visit_concept_id AS VARCHAR) AS visit_concept_ids, " \
              "  9223372036854775807 AS sort_order, " \
              "  observation_period_id, " \
              "  person.person_id " \
              "FROM visits " \
              "INNER JOIN person " \
              "  ON visits.person_id = person.person_id"
        con.execute(sql)
        if self._settings.has_labels:
            part1 = "  label, "
            part2 = "  INNER JOIN labels ON tokens.person_id = labels.person_id "
        else:
            part1 = ""
            part2 = ""
        sql = "SELECT tokens.*, " \
              f"{part1}" \
              "  ROW_NUMBER() OVER " \
              "    (PARTITION BY observation_period_id ORDER BY visit_concept_orders, sort_order) - 1  AS orders " \
              "FROM (" \
              "  SELECT * FROM interval_tokens " \
              "  UNION ALL " \
              "  SELECT * FROM start_tokens " \
              "  UNION ALL " \
              "  SELECT * FROM event_tokens " \
              "  UNION ALL " \
              "  SELECT * FROM end_tokens" \
              ") tokens " \
              f"{part2}" \
              "ORDER BY observation_period_id, visit_concept_orders, sort_order"
        union_tokens = con.execute(sql).arrow()
        con.execute("DROP TABLE visits")
        con.execute("DROP TABLE interval_tokens")
        con.execute("DROP TABLE start_tokens")
        con.execute("DROP TABLE event_tokens")
        con.execute("DROP TABLE end_tokens")
        duckdb.close(con)
        aggregate_list = [("person_id", "max"),
                          ("concept_ids", "list"),
                          ("visit_segments", "list"),
                          ("dates", "list"),
                          ("ages", "list"),
                          ("visit_concept_orders", "list"),
                          ("visit_concept_ids", "list"),
                          ("orders", "list"),
                          ("concept_ids", "count"),
                          ("visit_concept_orders", "max")]
        name_list = ["observation_period_id",
                     "person_id",
                     "concept_ids",
                     "visit_segments",
                     "dates",
                     "ages",
                     "visit_concept_orders",
                     "visit_concept_ids",
                     "orders",
                     "num_of_concepts",
                     "num_of_visits"]
        if self._settings.has_labels:
            aggregate_list.append(("label", "any"))
            name_list.append("label")
        sequence_data = union_tokens.group_by("observation_period_id"). \
            aggregate(aggregate_list). \
            rename_columns(name_list)
        return sequence_data


def main(args: List[str]):
    config = configparser.ConfigParser()
    with open(args[0]) as file:  # Explicitly opening file so error is thrown when not found
        config.read_file(file)
    settings = CdmProcessingSettings(config)
    cdm_data_processor = CdmDataProcessor(settings)
    cdm_data_processor.process_cdm_data()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise Exception("Must provide path to ini file as argument")
    else:
        main(sys.argv[1:])
