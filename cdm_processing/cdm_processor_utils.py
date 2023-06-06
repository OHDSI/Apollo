import os
from typing import List, Dict

import pyarrow.parquet as pq
import pyarrow.compute as pc
import pyarrow as pa
import duckdb

PERSON = "person"

PERSON_ID = "person_id"
LOGGER_FILE_NAME = "_cdm_processing_log.txt"  # Start with underscore so ignored by Parquet
PROFILE_MAX_PERSONS = 1000
START_DATE_FIELDS = {
    "observation_period": "observation_period_start_date",
    "visit_occurrence": "visit_start_date",
    "condition_occurrence": "condition_start_date",
    "drug_exposure": "drug_exposure_start_date",
    "procedure_occurrence": "procedure_date",
    "device_exposure": "device_exposure_start_date",
    "measurement": "measurement_date",
    "observation": "observation_date",
    "death": "death_date",
}
CONCEPT_ID_FIELDS = {
    "visit_occurrence": "visit_concept_id",
    "condition_occurrence": "condition_concept_id",
    "drug_exposure": "drug_concept_id",
    "procedure_occurrence": "procedure_concept_id",
    "device_exposure": "device_concept_id",
    "measurement": "measurement_concept_id",
    "observation": "observation_concept_id",
    "death": "death_concept_id",
}
START_DATE = "start_date"
CONCEPT_ID = "concept_id"
DOMAIN_TABLES = [
    "condition_occurrence",
    "drug_exposure",
    "procedure_occurrence",
    "device_exposure",
    "measurement",
    "observation",
    "death",
]
DEATH = "death"
DEATH_CONCEPT_ID = "death_concept_id"
DEATH_CONCEPT_ID_VALUE = 4306655
INTERNAL_VISIT_ID = "internal_visit_id"
VISIT_OCCURRENCE = "visit_occurrence"
VISIT_OCCURRENCE_ID = "visit_occurrence_id"
CONCEPT = "concept"
CONCEPT_ANCESTOR = "concept_ancestor"

def add_date_of_birth(person: pa.Table) -> pa.Table:
    """
    Compute the date of birth from a person table, and add it as a column called 'date_of_birth'.
    """
    dob = pc.strptime(
        pc.binary_join_element_wise(
            pc.cast(person['year_of_birth'], pa.string()),
            pc.cast(pc.coalesce(person['month_of_birth'], 1), pa.string()),
            pc.cast(pc.coalesce(person['day_of_birth'], 1), pa.string()),
            "-"
        ),
        format="%Y-%m-%d",
        unit="s"
    )
    return person.append_column("date_of_birth", dob)


def union_domain_tables(cdm_tables: Dict[str, pa.Table]) -> pa.Table:
    """
    Combines all domain tables into a single table. For this, column names will be normalized first.
    Entries in the death table will automatically be assigned a concept ID (4306655).

    Args:
        cdm_tables: A dictionary, mapping from CDM table name to table data.

    Returns:
        A table with a person_id, concept_id, visit_occurrence_id, and a start_date column.
    """
    available_domain_tables = list(set(DOMAIN_TABLES) & set(cdm_tables))
    normalized_tables = []
    if DEATH in cdm_tables:
        death = cdm_tables[DEATH]
        death = death.select([PERSON_ID, START_DATE_FIELDS[DEATH]]).rename_columns(
            [PERSON_ID, START_DATE]
        ).append_column(
            CONCEPT_ID,
            pa.array([DEATH_CONCEPT_ID_VALUE] * len(death), pa.int64())
        ).append_column(
            VISIT_OCCURRENCE_ID,
            pa.nulls(len(death), pa.int64())
        )
        available_domain_tables.remove(DEATH)
        normalized_tables.append(death)
    for table in available_domain_tables:
        normalized_table = cdm_tables[table].select(
            [PERSON_ID, START_DATE_FIELDS[table], CONCEPT_ID_FIELDS[table], VISIT_OCCURRENCE_ID]).rename_columns(
            [PERSON_ID, START_DATE, CONCEPT_ID, VISIT_OCCURRENCE_ID]
        )
        normalized_tables.append(normalized_table)
    event_table = pa.concat_tables(normalized_tables)
    return event_table


def remove_concepts(event_table: pa.Table, concept_ids: List[int]) -> tuple[pa.Table, int]:
    """
    Removes all rows from a table that have a concept ID in the given list.

    Args:
        event_table: A table with a person_id, concept_id and a start_date column.
        concept_ids: A list of concept IDs to remove.

    Returns:
        A tuple with the count of removed rows, and a table with the same schema as the input table, but without the
            rows with the given concept IDs.
    """
    count_before = len(event_table)
    result = event_table.filter(~pc.is_in(pc.field(CONCEPT_ID), pa.array(concept_ids)))
    return result, count_before - len(result)


def remove_duplicates(event_table: pa.Table) -> tuple[pa.Table, int]:
    """
    Removes all rows with duplicate person_id, concept_id, and start_date combinations.

    Args:
        event_table: A table with a person_id, concept_id and a start_date column.

    Returns:
        A tuple with the count of removed rows, and a table with the same schema as the input table, but without the
            duplicate rows.
    """
    count_before = len(event_table)
    con = duckdb.connect(database=':memory:', read_only=False)
    con.register("event_table", event_table)
    sql = "SELECT DISTINCT * FROM event_table"
    result = con.execute(sql).arrow()
    duckdb.close(con)
    return result, count_before - len(result)


def link_events_to_visits(event_table: pa.Table,
                          visit_occurrence: pa.Table,
                          mising_visit_concept_id: int = 0) -> tuple[pa.Table, pa.Table, Dict[str, int]]:
    """
    Links events to visits by finding the visit that contains the event's start date.

    Args:
        event_table: A table with a person_id, concept_id and a start_date column.
        visit_occurrence: The CDM visit_occurrence table
        mising_visit_concept_id: The concept ID to use for visits that were generated because events could not be
            linked to an existing visit.

    Returns:
        A tuple of 3 items: (1) the combined table with an additional internal_visit_id column, (2) the visit_occurrence
        table with an additional internal_visit_id column and added visits if some events could not be mapped to
        existing visits, and (3) a dataframe with the number of visits mapped by ID, data, or to new visits.
    """
    # DuckDb seems to be the fastest way (by far) to do these join, especially the one on dates
    visit_occurrence = visit_occurrence.append_column("internal_visit_id",
                                                      pa.array(range(len(visit_occurrence)), pa.int64()))
    con = duckdb.connect(database=':memory:', read_only=False)
    con.register("event_table", event_table)
    con.register("visit_occurrence", visit_occurrence)
    # Join by visit_occurrence_id
    sql = "CREATE TABLE joined_1 AS " \
          "SELECT event_table.*, " \
          "   internal_visit_id AS id_from_id " \
          "FROM event_table " \
          "LEFT JOIN visit_occurrence " \
          "  ON event_table.visit_occurrence_id = visit_occurrence.visit_occurrence_id "
    con.execute(sql)

    # Join by date
    sql = "CREATE TABLE joined_2 AS " \
          "SELECT joined_1.person_id, " \
          "  joined_1.start_date, " \
          "  joined_1.concept_id, " \
          "  joined_1.id_from_id, " \
          "  MIN(visit_occurrence.internal_visit_id) AS id_from_date " \
          "FROM joined_1 " \
          "LEFT JOIN visit_occurrence " \
          "  ON joined_1.person_id = visit_occurrence.person_id " \
          "    AND start_date >= visit_start_date " \
          "    AND start_date <= visit_end_date " \
          "GROUP BY joined_1.person_id, " \
          "  joined_1.start_date," \
          "  joined_1.concept_id, " \
          "  joined_1.id_from_id"
    con.execute(sql)

    # Create missing visits from unmapped event dates
    sql = "CREATE TABLE missing_visits AS " \
          "SELECT person_id, " \
          "  start_date AS visit_start_date, " \
          "  start_date AS visit_end_date," \
          "  {max_internal_visit_id} + ROW_NUMBER() OVER(ORDER BY person_id, start_date) AS internal_visit_id, " \
          "  CAST({mising_visit_concept_id} AS BIGINT) AS visit_concept_id " \
          "FROM ( " \
          "  SELECT DISTINCT person_id," \
          "    start_date " \
          "  FROM joined_2 " \
          "  WHERE id_from_id IS NULL" \
          "     AND id_from_date IS NULL " \
          ") AS missing_visits"
    sql = sql.format(max_internal_visit_id=len(visit_occurrence) - 1, mising_visit_concept_id=mising_visit_concept_id)
    con.execute(sql)
    new_visits = con.execute("SELECT COUNT(*) FROM missing_visits").fetchone()[0]
    existing_visits = len(visit_occurrence)
    # Join to the missing visits
    sql = "CREATE TABLE joined_3 AS " \
          "SELECT joined_2.*, " \
          "   missing_visits.internal_visit_id AS id_from_new_visit " \
          "FROM joined_2 " \
          "LEFT JOIN missing_visits " \
          "  ON joined_2.person_id = missing_visits.person_id " \
          "    AND start_date = visit_start_date"
    con.execute(sql)
    sql = "SELECT person_id, " \
          "  concept_id, " \
          "  start_date, " \
          "  COALESCE(id_from_id, id_from_date, id_from_new_visit) AS internal_visit_id " \
          "FROM joined_3"
    event_table = con.execute(sql).arrow()
    sql = "SELECT person_id, " \
          "  visit_start_date, " \
          "  visit_end_date, " \
          "  visit_concept_id, " \
          "  internal_visit_id " \
          "FROM visit_occurrence " \
          "" \
          "UNION ALL " \
          "" \
          "SELECT person_id, " \
          "  visit_start_date, " \
          "  visit_end_date, " \
          "  visit_concept_id, " \
          "  internal_visit_id " \
          "FROM missing_visits"
    visit_occurrence = con.execute(sql).arrow()

    sql = "SELECT CAST(SUM(mapped_by_id) AS INT) AS mapped_by_id, " \
          "  CAST(SUM(mapped_by_date) AS INT) AS mapped_by_date, " \
          "  CAST(SUM(mapped_by_new_visit) AS INT) AS mapped_to_new_visit " \
          "FROM ( " \
          "  SELECT CASE WHEN id_from_id IS NOT NULL THEN 1 ELSE 0 END AS mapped_by_id, " \
          "    CASE WHEN id_from_date IS NOT NULL AND id_from_id IS NULL THEN 1 ELSE 0 END AS mapped_by_date, " \
          "    CASE WHEN id_from_new_visit IS NOT NULL THEN 1 ELSE 0 END AS mapped_by_new_visit " \
          "  FROM joined_3 " \
          ") AS counts"
    statistics = con.execute(sql).fetchdf().iloc[0].to_dict()
    statistics["new_visits"] = new_visits
    statistics["existing_visits"] = existing_visits
    con.execute("DROP TABLE joined_1")
    con.execute("DROP TABLE joined_2")
    con.execute("DROP TABLE joined_3")
    con.execute("DROP TABLE missing_visits")
    duckdb.close(con)
    return event_table, visit_occurrence, statistics


def load_mapping_to_ingredients(cdm_folder: str) -> pa.Table:
    """
    Uses the concept and concept_ancestor table to construct a mapping from drugs to ingredients.
    Args:
        cdm_folder: The path where the CDM Parquet files are saved (using the GeneralPretrainModelTools packages).

    Returns:
        A dictionary from drug concept ID to ingredient concept ID.
    """
    ingredients = pq.read_table(
        os.path.join(cdm_folder, CONCEPT),
        columns=["concept_id"],
        filters=[("concept_class_id", "==", "Ingredient")],
    )
    concept_ancestor = pq.read_table(os.path.join(cdm_folder, CONCEPT_ANCESTOR))
    concept_ancestor = concept_ancestor.join(
        ingredients,
        keys=["ancestor_concept_id"],
        right_keys=["concept_id"],
        join_type="inner",
    ).select(["descendant_concept_id", "ancestor_concept_id"]).rename_columns(
        ["source_concept_id", "target_concept_id"])
    return concept_ancestor


def map_concepts(cdm_table: pa.Table, concept_id_field: str, mapping: pa.Table) -> pa.Table:
    """
    Maps a concept ID field to another concept ID using a mapping table.
    Args:
        cdm_table: The table to map.
        concept_id_field: The name of the field containing the concept ID.
        mapping: A table with two columns: source_concept_id and target_concept_id.

    Returns:
        A table with the same columns as cdm_table, but with the concept ID field replaced by the target_concept_id. Any
        records that did not have a matching concept were removed. Any records that map to multiple concepts are
        duplicated.
    """
    intermediate_columns = cdm_table.column_names
    intermediate_columns.remove(concept_id_field)
    intermediate_columns.append("target_concept_id")
    columns = cdm_table.column_names
    columns.remove(concept_id_field)
    columns.append(concept_id_field)
    return cdm_table.join(
        mapping,
        keys=[concept_id_field],
        right_keys=["source_concept_id"],
        join_type="inner",
    ).select(intermediate_columns).rename_columns(columns)


def add_observation_period_id(visit_occurrence: pa.Table, observation_period_table: pa.Table) -> pa.Table:
    """
    Adds the observation period ID to the visit occurrence table.
    Args:
        visit_occurrence: The visit occurrence table.
        observation_period_table: The observation period table.

    Returns:
        A table with the same columns as visit_occurrence, but with the observation period ID added.
    """
    con = duckdb.connect(database=':memory:', read_only=False)
    con.register("visit_occurrence", visit_occurrence)
    con.register("observation_period_table", observation_period_table)
    sql = "SELECT visit_occurrence.*, observation_period_table.observation_period_id " \
          "FROM visit_occurrence " \
          "INNER JOIN observation_period_table " \
          "  ON visit_occurrence.person_id = observation_period_table.person_id " \
          "    AND visit_occurrence.visit_start_date >= observation_period_table.observation_period_start_date " \
          "    AND visit_occurrence.visit_start_date <= observation_period_table.observation_period_end_date"
    result = con.execute(sql).arrow()
    duckdb.close(con)
    return result
