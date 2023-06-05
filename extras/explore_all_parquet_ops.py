import cdm_processing.cdm_arrow_processor_utils as cdm_utils

cdm_folder = "D:/GPM_CCAE"
cdm_tables = cdm_utils.get_cdm_tables(cdm_folder, 1)
dob = cdm_utils.compute_date_of_birth(cdm_tables["person"])
drug_mapping = cdm_utils.load_mapping_to_ingredients(cdm_folder=cdm_folder)
cdm_tables["drug_exposure"] = cdm_utils.map_concepts(cdm_table=cdm_tables["drug_exposure"],
                                                     concept_id_field="drug_concept_id",
                                                     mapping=drug_mapping)
combined_table = cdm_utils.union_domain_tables(cdm_tables)
combined_table, removed_concepts = cdm_utils.remove_concepts(combined_table=combined_table,
                                                             concept_ids=[0, 900000010])
combined_table, removed_duplicates = cdm_utils.remove_duplicates(combined_table=combined_table)
combined_table, visit_occurrence, mapping_stats = cdm_utils.link_events_to_visits(combined_table=combined_table,
                                                                                  visit_occurrence=cdm_tables["visit_occurrence"],
                                                                                  mising_visit_concept_id=1)
# TODO: join by obersvation period
x = combined_table.to_pandas()