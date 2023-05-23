from typing import Dict, List, Any
import datetime as dt

import pytest
import pandas as pd
import numpy as np

import cdm_processing.cdm_processor_utils as cpu


@pytest.fixture
def cdm_tables() -> Dict[str, pd.DataFrame]:
    person = pd.DataFrame(
        {
            "person_id": [1],
            "year_of_birth": [1970],
            "month_of_birth": [5],
            "day_of_birth": [7],
            "gender_concept_id": [8507],
        }
    )
    observation_period = pd.DataFrame(
        {
            "person_id": [1, 1, 1],
            "observation_period_id": [1, 2, 3],
            "observation_period_start_date": ["2000-01-01", "2001-01-01", "2002-01-01"],
            "observation_period_end_date": ["2000-07-01", "2001-07-01", "2002-07-01"],
        }
    )
    observation_period["observation_period_start_date"] = pd.to_datetime(
        observation_period["observation_period_start_date"]
    )
    observation_period["observation_period_end_date"] = pd.to_datetime(
        observation_period["observation_period_end_date"]
    )
    visit_occurrence = pd.DataFrame(
        {
            "person_id": [1, 1, 1],
            "visit_occurrence_id": [1, 2, 3],
            "visit_concept_id": [9201, 9202, 9201],
            "visit_start_date": ["2000-01-01", "2000-02-01", "2002-07-01"],
            "visit_end_date": ["2000-01-01", "2000-02-05", "2002-07-01"],
        }
    )
    visit_occurrence["visit_start_date"] = pd.to_datetime(
        visit_occurrence["visit_start_date"]
    )
    visit_occurrence["visit_end_date"] = pd.to_datetime(
        visit_occurrence["visit_end_date"]
    )
    condition_occurrence = pd.DataFrame(
        {
            "person_id": [1, 1, 1],
            "condition_concept_id": [123, 456, 0],
            "condition_start_date": ["2000-01-01", "2000-02-01", "2000-03-01"],
            "condition_end_date": ["2000-01-01", "2000-02-01", "2000-03-01"],
            "visit_occurrence_id": [1, np.NAN, np.NAN],
        }
    )
    condition_occurrence["condition_start_date"] = pd.to_datetime(
        condition_occurrence["condition_start_date"]
    )
    condition_occurrence["condition_end_date"] = pd.to_datetime(
        condition_occurrence["condition_end_date"]
    )
    death = pd.DataFrame(
        {
            "person_id": [1],
            "death_date": ["2020-07-01"],
        }
    )
    death["death_date"] = pd.to_datetime(
        death["death_date"]
    )
    cdm_tables = {"person": person,
                  "observation_period": observation_period,
                  "visit_occurrence": visit_occurrence,
                  "condition_occurrence": condition_occurrence,
                  "death": death}
    return cdm_tables


list_of_objects: List[Any] = []


def add_to_list(observation_period: pd.Series, cdm_tables: Dict[str, pd.DataFrame]):
    list_of_objects.append(cdm_tables)


def test_call_per_observation_period(cdm_tables: Dict[str, pd.DataFrame]):
    list_of_objects.clear()
    cpu.call_per_observation_period(cdm_tables, add_to_list)
    assert len(list_of_objects) == 3
    visits = list_of_objects[0]["visit_occurrence"]
    assert len(visits) == 2
    visits = list_of_objects[1]["visit_occurrence"]
    assert len(visits) == 0
    visits = list_of_objects[2]["visit_occurrence"]
    assert len(visits) == 1

def test_remove_concepts(cdm_tables: Dict[str, pd.DataFrame]):
    new_cdm_tables, removed_counts = cpu.remove_concepts(cdm_tables)
    assert removed_counts["condition_occurrence"] == 1
    assert len(new_cdm_tables["condition_occurrence"] == 2)


def test_union_domain_tables(cdm_tables: Dict[str, pd.DataFrame]):
    unioned_tables = cpu.union_domain_tables(cdm_tables)

    # Last record should be death:
    record = unioned_tables.iloc[-1]
    assert record["concept_id"] == 4306655


def test_get_date_of_birth():
    person = pd.Series(
        {
            "person_id": 1,
            "year_of_birth": 1970,
            "month_of_birth": 5,
            "day_of_birth": 7,
            "gender_concept_id": 8507,
        }
    )
    dob = cpu.get_date_of_birth(person)
    assert dob == dt.datetime(1970, 5, 7).date()

    person = pd.Series(
        {
            "person_id": 1,
            "year_of_birth": 1980,
            "month_of_birth": 4,
            "day_of_birth": np.NAN,
            "gender_concept_id": 8507,
        }
    )
    dob = cpu.get_date_of_birth(person)
    assert dob == dt.datetime(1980, 4, 1).date()

    person = pd.Series(
        {
            "person_id": 1,
            "year_of_birth": 1990,
            "month_of_birth": np.NAN,
            "day_of_birth": np.NAN,
            "gender_concept_id": 8507,
        }
    )
    dob = cpu.get_date_of_birth(person)
    assert dob == dt.datetime(1990, 1, 1).date()


def test_group_by_visit(cdm_tables: Dict[str, pd.DataFrame]):
    visit_groups = cpu.group_by_visit(cdm_tables, link_by_date=True, create_missing_visits=True)
    assert len(visit_groups) == 5
    visit_group = visit_groups[0]  # First visit in CDM data, linked by ID
    assert len(visit_group.cdm_tables["condition_occurrence"]) == 1
    assert visit_group.cdm_tables["condition_occurrence"]["condition_concept_id"].iat[0] == 123
    visit_group = visit_groups[1]  # Second visit in CDM data, linked by date
    assert len(visit_group.cdm_tables["condition_occurrence"]) == 1
    assert visit_group.cdm_tables["condition_occurrence"]["condition_concept_id"].iat[0] == 456
    visit_group = visit_groups[2]  # New visit, derived from condition occurrence
    assert visit_group.visit["visit_concept_id"] == 1
    assert len(visit_group.cdm_tables["condition_occurrence"]) == 1
    assert visit_group.cdm_tables["condition_occurrence"]["condition_concept_id"].iat[0] == 0
    visit_group = visit_groups[3]  # Third visit in CDM data, linked by date
    assert "condition_occurrence" not in visit_group.cdm_tables
    visit_group = visit_groups[4]  # New visit, derived from death
    assert visit_group.visit["visit_concept_id"] == 1
    assert len(visit_groups[4].cdm_tables["death"]) == 1

    visit_groups = cpu.group_by_visit(cdm_tables, link_by_date=True, create_missing_visits=False)
    assert len(visit_groups) == 3
    visit_group = visit_groups[0]  # First visit in CDM data, linked by ID
    assert len(visit_group.cdm_tables["condition_occurrence"]) == 1
    assert visit_group.cdm_tables["condition_occurrence"]["condition_concept_id"].iat[0] == 123
    visit_group = visit_groups[1]  # Second visit in CDM data, linked by date
    assert len(visit_group.cdm_tables["condition_occurrence"]) == 1
    assert visit_group.cdm_tables["condition_occurrence"]["condition_concept_id"].iat[0] == 456
    visit_group = visit_groups[2]  # Third visit in CDM data, linked by date
    assert "condition_occurrence" not in visit_group.cdm_tables

    visit_groups = cpu.group_by_visit(cdm_tables, link_by_date=False, create_missing_visits=False)
    assert len(visit_groups) == 3
    visit_group = visit_groups[1]
    assert "condition_occurrence" not in visit_group.cdm_tables
    visit_group = visit_groups[2]
    assert "condition_occurrence" not in visit_group.cdm_tables
