from typing import Dict, List, Callable
import datetime as dt
import os

import pandas as pd
import numpy as np
import pyarrow.parquet as pq

OBSERVATION_PERIOD = "observation_period"
OBSERVATION_PERIOD_ID = "observation_period_id"
OBSERVATION_PERIOD_START_DATE = "observation_period_start_date"
OBSERVATION_PERIOD_END_DATE = "observation_period_end_date"
PERSON_ID = "person_id"
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
    "death": "death_date",
}
START_DATE = "start_date"
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
DEATH_CONCEPT_ID = 4306655
CONCEPT_ID = "concept_id"
YEAR_OF_BIRTH = "year_of_birth"
MONTH_OF_BIRTH = "month_of_birth"
DAY_OF_BIRTH = "day_of_birth"
VISIT_OCCURRENCE = "visit_occurrence"
VISIT_OCCURRENCE_ID = "visit_occurrence_id"
VISIT_START_DATE = "visit_start_date"
VISIT_END_DATE = "visit_end_date"
VISIT_CONCEPT_ID = "visit_concept_id"
CONCEPT = "concept"
CONCEPT_ANCESTOR = "concept_ancestor"
DRUG_CONCEPT_ID = "drug_concept_id"
CONCEPT_RELATIONSHIP = "concept_relationship"
CLASS_IDS_3_DIGITS = [
    "3-char nonbill code",
    "3-dig nonbill code",
    "3-char billing code",
    "3-dig billing code",
    "3-dig billing E code",
    "3-dig billing V code",
    "3-dig nonbill E code",
    "3-dig nonbill V code",
]


def call_per_observation_period(
        cdm_tables: Dict[str, pd.DataFrame],
        function: Callable[[pd.Series, Dict[str, pd.DataFrame]], None],
):
    """
    Calls the provided function for each observation period. CDM tables are filtered to only those events
    that fall in the observation period. 

    Args:
        cdm_tables: A dictionary, mapping from CDM table name to table data.
        function: The function to call for each observation period.The function should have two arguments:
                  the observation_period (Series), and a dictionary of CDM tables.
    """
    for index, observation_period in cdm_tables[OBSERVATION_PERIOD].iterrows():
        observation_period_start_date = observation_period[
            OBSERVATION_PERIOD_START_DATE
        ]
        observation_period_end_date = observation_period[OBSERVATION_PERIOD_END_DATE]
        new_cdm_tables = {}
        for table_name, table in cdm_tables.items():
            if table_name in START_DATE_FIELDS:
                start_dates = table[START_DATE_FIELDS[table_name]]
                table = table[
                    (start_dates >= observation_period_start_date)
                    & (start_dates <= observation_period_end_date)
                    ].copy()
            new_cdm_tables[table_name] = table
        function(observation_period, new_cdm_tables)


def remove_concepts(cdm_tables: Dict[str, pd.DataFrame], concept_ids: List[int]) -> (Dict[str, pd.DataFrame],
                                                                                     Dict[str, int]):
    """
    Removes rows from the CDM tables where the concept ID is in the list of provided concept IDs.
    
    Args:
        cdm_tables: A dictionary, mapping from CDM table name to table data.
        concept_ids: A list of concept IDs to remove.

    Returns:
        The CDM tables with rows containing the concept IDs removed, and a dictionary containing the number of rows
        removed per table.
    """
    new_cdm_tables: Dict[str, pd.DataFrame] = {}
    removed_row_counts: Dict[str, int] = {}
    for table_name in cdm_tables:
        cdm_table = cdm_tables[table_name]
        if table_name in CONCEPT_ID_FIELDS:
            idx = [x not in concept_ids for x in cdm_table[CONCEPT_ID_FIELDS[table_name]]]
            removed_row_counts[table_name] = len(idx) - sum(idx)
            cdm_table = cdm_table[idx].copy()
        new_cdm_tables[table_name] = cdm_table
    return new_cdm_tables, removed_row_counts


def union_domain_tables(cdm_tables: Dict[str, pd.DataFrame], include_person_id=False) -> pd.DataFrame:
    """
    Combines all domain tables into a single table. For this, column names will be normalized first.
    Entries in the death table will automatically be assigned a concept ID (4306655).

    Args:
        cdm_tables: A dictionary, mapping from CDM table name to table data.
        include_person_id: Include the person_id column in the results?

    Returns:
        A data frame with a concept_id and a start_date column, and if requested a person_id column. The result will
        be sorted by start date and then concept ID.
    """
    # Using numpy arrays for massive boost in speed (compared to pandas data frames):
    concept_ids = []
    start_dates = []
    person_ids = []
    for table_name in DOMAIN_TABLES:
        if table_name in cdm_tables:
            table = cdm_tables[table_name]
            # table.reset_index(drop=True, inplace=True)
            if table_name == DEATH:
                concept_ids.append(np.asarray([DEATH_CONCEPT_ID] * len(table)))
                start_dates.append(table[START_DATE_FIELDS[table_name]].to_numpy())
            else:
                concept_ids.append(table[CONCEPT_ID_FIELDS[table_name]].to_numpy())
                start_dates.append(table[START_DATE_FIELDS[table_name]].to_numpy())
            if include_person_id:
                person_ids.append(table[PERSON_ID].to_numpy())
    if len(concept_ids) == 0:
        if include_person_id:
            return pd.DataFrame({CONCEPT_ID: [], START_DATE: [], PERSON_ID: []})
        else:
            return pd.DataFrame({CONCEPT_ID: [], START_DATE: []})
    else:
        concept_ids = np.concatenate(concept_ids)
        start_dates = np.concatenate(start_dates)
        sorted_idx = np.lexsort((concept_ids, start_dates))
        if include_person_id:
            person_ids = np.concatenate(person_ids)
            result = pd.DataFrame(
                {CONCEPT_ID: concept_ids[sorted_idx], START_DATE: start_dates[sorted_idx],
                 PERSON_ID: person_ids[sorted_idx]})
        else:
            result = pd.DataFrame({CONCEPT_ID: concept_ids[sorted_idx], START_DATE: start_dates[sorted_idx]})
        return result


def get_date_of_birth(person: pd.Series) -> dt.date:
    """
    Computes a date of birth from a person entry

    Args:
        person: A single row from the person table.
    """
    year = person[YEAR_OF_BIRTH]
    month = person[MONTH_OF_BIRTH]
    day = person[DAY_OF_BIRTH]
    if pd.isna(month):
        month = 1
    if pd.isna(day):
        day = 1
    return dt.date(year=int(year), month=int(month), day=int(day))


class VisitData:
    """
    Class for grouping all CDM data for one visit.
    """

    visit: pd.Series
    visit_start_date: dt.date
    cdm_tables: Dict[str, pd.DataFrame]
    mapped_by_id: int
    mapped_by_date: int
    mapped_to_new_visit: int
    new_visit: bool

    def __init__(self, visit: pd.Series, new_visit: bool = False):
        self.visit = visit
        self.cdm_tables = {}
        self.visit_start_date = visit[VISIT_START_DATE]
        self.mapped_by_id = 0
        self.mapped_by_date = 0
        self.mapped_to_new_visit = 0
        self.new_visit = new_visit


def group_by_visit(
        cdm_tables: Dict[str, pd.DataFrame],
        link_by_date: bool = True,
        create_missing_visits: bool = True,
        missing_visit_concept_id: int = 1,
) -> List[VisitData]:
    """
    Groups events by visit.

    Args:
        cdm_tables: A dictionary, mapping from CDM table name to table data.
        link_by_date: If true, events not linked to an existing visit by visit_occurrence_id
                      will be linked to an existing visit if the event date falls within the
                      visit start and end date.
        create_missing_visits: If no visit exists with dates corresponding to an event, a new
                               one-day visit will be created.
        missing_visit_concept_id: The visit_concept_id to be used for newly created visits if
                                  create_missing_visits is true.

    Yields:
        A list of type VisitData, sorted by visit start date.
    """
    if VISIT_OCCURRENCE in cdm_tables:
        visits = cdm_tables[VISIT_OCCURRENCE]
    else:
        visits = pd.DataFrame()
    visit_indices = list(range(len(visits)))
    visit_datas = [VisitData(visits.iloc[i]) for i in range(len(visits))]
    for table_name in DOMAIN_TABLES:
        if table_name in cdm_tables:
            cdm_table = cdm_tables[table_name]
            if len(cdm_table) == 0:
                continue
            start_date_field = START_DATE_FIELDS[table_name]
            if len(visits) == 0:
                event_visit_index = np.empty(shape=len(cdm_table), dtype=np.int32)
                event_visit_index.fill(-1)
            else:
                if VISIT_OCCURRENCE_ID in cdm_table:
                    event_visit_index = np.piecewise(
                        np.zeros(len(cdm_table), dtype=int),
                        [
                            (
                                    cdm_table[VISIT_OCCURRENCE_ID].values
                                    == visit_occurrence_id
                            )
                            for visit_occurrence_id in zip(visits[VISIT_OCCURRENCE_ID].values)
                        ],
                        np.append(visit_indices, -1),
                    )
                    index, count = np.unique(event_visit_index, return_counts=True)
                    for i in range(len(index)):
                        if index[i] != -1:
                            visit_datas[index[i]].mapped_by_id += count[i]
                else:
                    event_visit_index = np.empty(shape=len(cdm_table), dtype=np.int32)
                    event_visit_index.fill(-1)

                if link_by_date:
                    idx = event_visit_index == -1
                    if any(idx):
                        event_visit_index[idx] = np.piecewise(
                            np.zeros(sum(idx), dtype=int),
                            [
                                (cdm_table.loc[idx, start_date_field].values >= start_date)
                                & (cdm_table.loc[idx, start_date_field].values <= end_date)
                                for start_date, end_date in zip(
                                    visits[VISIT_START_DATE].values,
                                    visits[VISIT_END_DATE].values,
                                )
                            ],
                            np.append(visit_indices, -1),
                        )
                        index, count = np.unique(event_visit_index[idx], return_counts=True)
                        for i in range(len(index)):
                            if index[i] != -1:
                                visit_datas[index[i]].mapped_by_date += count[i]
            if create_missing_visits:
                idx = event_visit_index == -1
                if any(idx):
                    dates = cdm_table.loc[idx, start_date_field].unique()
                    person_id = cdm_table[PERSON_ID].iat[0]
                    missing_visit_indices = list(
                        range(len(visits), len(visits) + len(dates))
                    )
                    missing_visits = pd.DataFrame(
                        {
                            PERSON_ID: [person_id] * len(dates),
                            VISIT_OCCURRENCE_ID: [np.NAN] * len(dates),
                            VISIT_CONCEPT_ID: [missing_visit_concept_id] * len(dates),
                            VISIT_START_DATE: dates,
                            VISIT_END_DATE: dates,
                        }
                    )
                    event_visit_index[idx] = np.piecewise(
                        [0] * sum(idx),
                        [
                            (cdm_table.loc[idx, start_date_field].values == start_date)
                            for start_date in zip(missing_visits[VISIT_START_DATE].values)
                        ],
                        missing_visit_indices,
                    )
                    visits = pd.concat([visits, missing_visits])
                    visit_indices.extend(missing_visit_indices)
                    visit_datas += [
                        VisitData(missing_visits.iloc[i], new_visit=True)
                        for i in range(len(missing_visits))
                    ]
                    index, count = np.unique(missing_visit_indices, return_counts=True)
                    for i in range(len(index)):
                        visit_datas[index[i]].mapped_to_new_visit += count[i]
            else:
                idx = event_visit_index != -1
                cdm_table = cdm_table[idx]
                event_visit_index = event_visit_index[idx]

            for visit_index, events in cdm_table.groupby(event_visit_index):
                visit_datas[visit_index].cdm_tables[table_name] = events
    visit_datas.sort(key=lambda x: x.visit_start_date)
    return visit_datas


def load_mapping_to_ingredients(cdm_data_path: str) -> Dict[int, int]:
    """
    Uses the concept and concept_ancestor table to construct a mapping from drugs to ingredients.
    Args:
        cdm_data_path: The path where the CDM Parquet files are saved (using the GeneralPretrainModelTools packages).

    Returns:
        A dictionary from drug concept ID to ingredient concept ID.
    """
    ingredients = pq.read_table(
        os.path.join(cdm_data_path, CONCEPT),
        columns=["concept_id"],
        filters=[("concept_class_id", "==", "Ingredient")],
    )
    concept_ancestor = pq.read_table(os.path.join(cdm_data_path, CONCEPT_ANCESTOR))
    concept_ancestor = concept_ancestor.join(
        ingredients,
        keys=["ancestor_concept_id"],
        right_keys=["concept_id"],
        join_type="inner",
    )
    mapping = pd.DataFrame(concept_ancestor.to_pandas())
    mapping = dict(zip(mapping["descendant_concept_id"], mapping["ancestor_concept_id"]))
    return mapping


# Note: this does not appear to be a good idea. E.g. 'Brain injury without open intracranial wound' was mapped
# to 'Disorder of nervous system', which seems a bit too generic.
def load_mapping_to_3_digit_condition_codes(cdm_data_path: str) -> Dict[int, int]:
    """
    Uses the concept and concept_ancestor table to construct a mapping from conditions to concepts equivalent to 3-digit
    ICD-9 or ICD-10 codes.
    Args:
        cdm_data_path: The path where the CDM Parquet files are saved (using the GeneralPretrainModelTools packages).

    Returns:
        A dictionairy from condition concept ID to concept IDs representing 3-digit ICD-9 and 10 codes.
    """
    three_digit_concepts = pq.read_table(
        os.path.join(cdm_data_path, CONCEPT),
        columns=["concept_id"],
        filters=[("concept_class_id", "in", CLASS_IDS_3_DIGITS)],
    )
    concept_relationship = pq.read_table(
        os.path.join(cdm_data_path, CONCEPT_RELATIONSHIP),
        filters=[("relationship_id", "==", "Maps to")],
    )
    concept_relationship = concept_relationship.join(
        three_digit_concepts,
        keys=["concept_id_1"],
        right_keys=["concept_id"],
        join_type="inner",
    )
    concept_ancestor = pq.read_table(os.path.join(cdm_data_path, CONCEPT_ANCESTOR))
    concept_ancestor = concept_ancestor.join(
        concept_relationship,
        keys=["ancestor_concept_id"],
        right_keys=["concept_id_2"],
        join_type="inner",
    )
    mapping = pd.DataFrame(
        concept_ancestor.select(
            ["ancestor_concept_id", "descendant_concept_id"]
        ).to_pandas()
    )
    mapping = dict(zip(mapping["descendant_concept_id"], mapping["ancestor_concept_id"]))
    return mapping


def map_concepts(cdm_table: pd.DataFrame, concept_id_field: str, mapping: Dict[int, int]) -> pd.DataFrame:
    """
    Map drugs to ingredients.

    Args:
        cdm_table: A data frame with records from a CDM table.
        concept_id_field: The name of the concept ID field to be mapped.
        mapping: The dictionary as generated using one of the load_mapping_ functions.

    Returns:
        The CDM table, with the concept ID replaced with the mapped concept IDs. Any records that
        did not have a matching concept were removed. Any records that map to multiple concepts are duplicated.
    """

    def do_map(x):
        if x in mapping:
            return mapping[x]
        else:
            return -1

    mapped_ids = [do_map(x) for x in cdm_table[concept_id_field]]
    cdm_table[concept_id_field] = mapped_ids
    cdm_table = cdm_table[cdm_table[concept_id_field] != -1]
    return cdm_table
