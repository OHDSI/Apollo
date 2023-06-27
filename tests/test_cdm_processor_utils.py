from typing import List

import unittest
import pyarrow as pa
import pyarrow.compute as pc
import datetime as dt

import cdm_processing.cdm_processor_utils as cdm_utils


def d(dates_as_strings: List[str]):
    """
    Helper function: convert a list of dates as strings to a list of dates as datetime objects.
    Args:
        dates_as_strings: A list of dates as strings in the format YYYY-MM-DD.

    Returns:
        A list of dates as datetime objects.
    """
    return pc.strptime(
        dates_as_strings,
        format="%Y-%m-%d",
        unit="s"
    )


class TestCdmProcessorUtils(unittest.TestCase):

    def setUp(self) -> None:
        person = pa.Table.from_pydict(
            {
                "person_id": [1],
                "year_of_birth": [1970],
                "month_of_birth": [5],
                "day_of_birth": [7],
                "gender_concept_id": [8507],
            },
            schema=pa.schema(
                [
                    ("person_id", pa.int64()),
                    ("year_of_birth", pa.int32()),
                    ("month_of_birth", pa.int32()),
                    ("day_of_birth", pa.int32()),
                    ("gender_concept_id", pa.int32()),
                ]
            )
        )
        observation_period = pa.Table.from_pydict(
            {
                "person_id": [1, 1, 1],
                "observation_period_id": [1, 2, 3],
                "observation_period_start_date": d(["2000-01-01", "2001-01-01", "2002-01-01"]),
                "observation_period_end_date": d(["2000-07-01", "2001-07-01", "2002-07-01"]),
            },
            schema=pa.schema(
                [
                    ("person_id", pa.int64()),
                    ("observation_period_id", pa.int64()),
                    ("observation_period_start_date", pa.date32()),
                    ("observation_period_end_date", pa.date32()),
                ]
            )
        )
        visit_occurrence = pa.Table.from_pydict(
            {
                "person_id": pa.array([1, 1, 1]),
                "visit_occurrence_id": [1, 2, 3],
                "visit_concept_id": [9201, 9202, 9201],
                "visit_start_date": d(["2000-01-01", "2000-02-01", "2002-07-01"]),
                "visit_end_date": d(["2000-01-01", "2000-02-05", "2002-07-01"]),
            },
            schema=pa.schema(
                [
                    ("person_id", pa.int64()),
                    ("visit_occurrence_id", pa.int64()),
                    ("visit_concept_id", pa.int32()),
                    ("visit_start_date", pa.date32()),
                    ("visit_end_date", pa.date32()),
                ]
            )
        )
        condition_occurrence = pa.Table.from_pydict(
            {
                "person_id": [1, 1, 1],
                "condition_concept_id": [123, 456, 0],
                "condition_start_date": d(["2000-01-01", "2000-02-01", "2000-03-01"]),
                "condition_end_date": d(["2000-01-01", "2000-02-01", "2000-03-01"]),
                "visit_occurrence_id": [1, None, None],
            },
            schema=pa.schema(
                [
                    ("person_id", pa.int64()),
                    ("condition_concept_id", pa.int32()),
                    ("condition_start_date", pa.date32()),
                    ("condition_end_date", pa.date32()),
                    ("visit_occurrence_id", pa.int64()),
                ]
            )
        )
        death = pa.Table.from_pydict(
            {
                "person_id": [1],
                "death_date": d(["2020-07-01"]),
            },
            schema=pa.schema(
                [
                    ("person_id", pa.int64()),
                    ("death_date", pa.date32()),
                ]
            )
        )
        self.cdm_tables = {"person": person,
                           "observation_period": observation_period,
                           "visit_occurrence": visit_occurrence,
                           "condition_occurrence": condition_occurrence,
                           "death": death}

    def test_union_domain_tables(self):
        event_table = cdm_utils.union_domain_tables(self.cdm_tables)

        # First record should be death:
        assert event_table["concept_id"].to_pylist()[0] == 4306655

    def test_remove_concepts(self):
        event_table = cdm_utils.union_domain_tables(self.cdm_tables)
        new_cdm_tables, removed_count = cdm_utils.remove_concepts(event_table=event_table, concept_ids=[0])
        assert removed_count == 1
        assert len(event_table) == 4

    def test_add_date_of_birth(self):
        person = pa.Table.from_pydict(
            {
                "person_id": [1, 2, 3],
                "year_of_birth": [1970, 1980, 1990],
                "month_of_birth": [5, 4, None],
                "day_of_birth": [7, None, None],
                "gender_concept_id": [8507, 8507, 8507],
            }
        )
        dob = cdm_utils.add_date_of_birth(person)
        assert dob["date_of_birth"].to_pylist()[0] == dt.datetime(1970, 5, 7)
        assert dob["date_of_birth"].to_pylist()[1] == dt.datetime(1980, 4, 1)
        assert dob["date_of_birth"].to_pylist()[2] == dt.datetime(1990, 1, 1)

    def test_group_by_visit(self):
        event_table = cdm_utils.union_domain_tables(self.cdm_tables)
        visits = self.cdm_tables["visit_occurrence"]
        event_table, visits, stats = cdm_utils.link_events_to_visits(event_table=event_table,
                                                                     visit_occurrence=visits,
                                                                     mising_visit_concept_id=0)
        assert pc.max(visits["internal_visit_id"]).as_py() == 4
        # First visit in CDM data, linked by ID:
        visit_group = event_table.filter(pc.equal(event_table["internal_visit_id"], 0))
        assert len(visit_group) == 1
        assert visit_group["concept_id"].to_pylist()[0] == 123
        # Second visit in CDM data, linked by date
        visit_group = event_table.filter(pc.equal(event_table["internal_visit_id"], 1))
        assert len(visit_group) == 1
        assert visit_group["concept_id"].to_pylist()[0] == 456
        # Third visit in CDM data, has no events:
        visit_group = event_table.filter(pc.equal(event_table["internal_visit_id"], 2))
        assert len(visit_group) == 0
        # New visit, derived from condition occurrence
        visit_group = event_table.filter(pc.equal(event_table["internal_visit_id"], 3))
        assert len(visit_group) == 1
        assert visit_group["concept_id"].to_pylist()[0] == 0
        visit = visits.filter(pc.equal(visits["internal_visit_id"], 3))
        assert visit["visit_concept_id"].to_pylist()[0] == 0
        # New visit, derived from death
        visit_group = event_table.filter(pc.equal(event_table["internal_visit_id"], 4))
        assert len(visit_group) == 1
        assert visit_group["concept_id"].to_pylist()[0] == 4306655

    def test_remove_duplicate_events(self):
        event_table = pa.Table.from_pydict(
            {
                "concept_id": [1000, 2000, 2000, 2000],
                "start_date": d(["2000-01-01", "2001-01-01", "2001-01-01", "2001-01-02"]),
            }
        )
        event_table, removed_count = cdm_utils.remove_duplicates(event_table)
        assert removed_count == 1
        assert len(event_table) == 3
        assert event_table["concept_id"].to_pylist() == [1000, 2000, 2000]


if __name__ == '__main__':
    unittest.main()
