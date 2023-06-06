import cdm_processing.cdm_arrow_processor_utils as cdm_utils

cdm_folder = "D:/GPM_CCAE"
cdm_tables = cdm_utils.get_cdm_tables(cdm_folder, 1)
cdm_tables["person"] = cdm_utils.add_date_of_birth(cdm_tables["person"])
drug_mapping = cdm_utils.load_mapping_to_ingredients(cdm_folder=cdm_folder)
cdm_tables["drug_exposure"] = cdm_utils.map_concepts(cdm_table=cdm_tables["drug_exposure"],
                                                     concept_id_field="drug_concept_id",
                                                     mapping=drug_mapping)
combined_table = cdm_utils.union_domain_tables(cdm_tables)
combined_table, removed_concepts = cdm_utils.remove_concepts(combined_table=combined_table,
                                                             concept_ids=[0, 900000010])
combined_table, removed_duplicates = cdm_utils.remove_duplicates(combined_table=combined_table)
combined_table, visit_occurrence, mapping_stats = cdm_utils.link_events_to_visits(combined_table=combined_table,
                                                                                  visit_occurrence=cdm_tables[
                                                                                      "visit_occurrence"],
                                                                                  mising_visit_concept_id=1)

# Create CEHR-BERT sequence format
import duckdb

con = duckdb.connect(database=':memory:', read_only=False)
con.register("visit_occurrence", visit_occurrence)
con.register("observation_period_table", cdm_tables["observation_period"])
con.register("person", cdm_tables["person"])
con.register("combined_table", combined_table)
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
      "  END AS concept_id, " \
      "  0 AS visit_segments, " \
      "  0 AS dates, " \
      " -1 AS ages, " \
      "  visit_rank AS visit_concept_orders, " \
      "  0 AS visit_concept_ids, " \
      "  -2 AS sort_order, " \
      "  observation_period_id " \
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
      "SELECT 'VS' AS concept_id, " \
      "  visit_rank % 2 + 1 AS visit_segments, " \
      "  DATE_DIFF('week', DATE '1970-01-01', visit_start_date) AS dates, " \
      "  DATE_DIFF('month', date_of_birth, visit_start_date) AS ages, " \
      "  visit_rank AS visit_concept_orders, " \
      "  visit_concept_id AS visit_concept_ids, " \
      "  -1 AS sort_order, " \
      "  observation_period_id " \
      "FROM visits " \
      "INNER JOIN person " \
      "  ON visits.person_id = person.person_id"
con.execute(sql)

sql = "CREATE TABLE event_tokens AS " \
      "SELECT CAST(concept_id AS VARCHAR) AS concept_id, " \
      "  visit_rank % 2 + 1 AS visit_segments, " \
      "  DATE_DIFF('week', DATE '1970-01-01', start_date) AS dates, " \
      "  DATE_DIFF('month', date_of_birth,start_date) AS ages, " \
      "  visit_rank AS visit_concept_orders, " \
      "  visit_concept_id AS visit_concept_ids, " \
      "  concept_id AS sort_order, " \
      "  observation_period_id " \
      "FROM combined_table " \
      "INNER JOIN visits " \
      "  ON combined_table.internal_visit_id = visits.internal_visit_id " \
      "INNER JOIN person " \
      "  ON visits.person_id = person.person_id"
con.execute(sql)

sql = "CREATE TABLE end_tokens AS " \
      "SELECT 'VE' AS concept_id, " \
      "  visit_rank % 2 + 1 AS visit_segments, " \
      "  DATE_DIFF('week', DATE '1970-01-01', visit_end_date) AS dates, " \
      "  DATE_DIFF('month', date_of_birth, visit_end_date) AS ages, " \
      "  visit_rank AS visit_concept_orders, " \
      "  visit_concept_id AS visit_concept_ids, " \
      "  9223372036854775807 AS sort_order, " \
      "  observation_period_id " \
      "FROM visits " \
      "INNER JOIN person " \
      "  ON visits.person_id = person.person_id"
con.execute(sql)

sql = "SELECT *, " \
      "  ROW_NUMBER() OVER (PARTITION BY observation_period_id ORDER BY sort_order) AS orders " \
      "FROM (" \
      "  SELECT * FROM interval_tokens " \
      "  UNION ALL " \
      "  SELECT * FROM start_tokens " \
      "  UNION ALL " \
      "  SELECT * FROM event_tokens " \
      "  UNION ALL " \
      "  SELECT * FROM end_tokens" \
      ") tokens " \
      "ORDER BY observation_period_id, visit_concept_orders, sort_order"
union_tokens = con.execute(sql).arrow()
con.execute("DROP TABLE interval_tokens")
con.execute("DROP TABLE start_tokens")
con.execute("DROP TABLE event_tokens")
con.execute("DROP TABLE end_tokens")

cehr_bert_input = union_tokens.group_by("observation_period_id").aggregate(
    [("concept_id", "list"), ("visit_segments", "list"), ("dates", "list"), ("ages", "list"),
     ("visit_concept_orders", "list"), ("visit_concept_ids", "list"), ("orders", "list"), ("concept_id", "count"),
     ("visit_concept_orders", "max")]).rename_columns(["observation_period_id", "concept_id", "visit_segments", "dates",
                                                       "ages", "visit_concept_orders", "visit_concept_ids", "orders",
                                                       "num_of_concepts", "num_of_visits"])
