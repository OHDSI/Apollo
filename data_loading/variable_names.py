class DataNames:
    """
    Names of the various fields in the input data
    """
    CONCEPT_IDS = "concept_ids"
    VISIT_SEGMENTS = "visit_segments"
    DATES = "dates"
    AGES = "ages"
    VISIT_CONCEPT_ORDERS = "visit_concept_orders"
    VISIT_CONCEPT_IDS = "visit_concept_ids"
    ORDERS = "orders"
    NUM_OF_CONCEPTS = "num_of_concepts"
    NUM_OF_VISITS = "num_of_visits"
    LABEL = "label"


class ModelInputNames:
    """
    Names of the inputs to the model. These inputs are generated by the data generator using the learning objectives.
    """
    LABEL = "label"
    MASKED_VISIT_CONCEPTS = "masked_visit_concepts"
    MASK_VISIT = "mask_visit"
    VISIT_PREDICTIONS = "visit_predictions"
    MASKED_TOKEN_IDS = "masked_token_ids"
    TOKEN_IDS = "token_ids"
    PADDING_MASK = "padding_mask"
    MASKED_TOKEN_MASK = "masked_token_mask"
    DATES = "dates"
    VISIT_SEGMENTS = "visit_segments"
    AGES = "ages"
    VISIT_CONCEPT_ORDERS = "visit_concept_orders"
    TOKEN_PREDICTIONS = "token_predictions"