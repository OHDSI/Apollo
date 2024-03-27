from dataclasses import dataclass


@dataclass
class ModelSettings:
    max_sequence_length: int
    concept_embedding: bool
    visit_order_embedding: bool
    segment_embedding: bool
    age_embedding: bool
    date_embedding: bool
    visit_concept_embedding: bool
    hidden_size: int
    num_attention_heads: int
    num_hidden_layers: int
    intermediate_size: int
    hidden_act: str
    embedding_combination_method: str
    hidden_dropout_prob: float
    attention_probs_dropout_prob: float

    def __post_init__(self):
        if self.hidden_act not in ["relu", "gelu"]:
            raise ValueError(f"Invalid hidden_act: {self.hidden_act}")
        if self.embedding_combination_method not in ["concat", "sum"]:
            raise ValueError(f"Invalid embedding_combination_method: {self.embedding_combination_method}")


@dataclass
class SimpleModelSettings:
    max_sequence_length: int
    concept_embedding: bool
    visit_order_embedding: bool
    segment_embedding: bool
    age_embedding: bool
    date_embedding: bool
    visit_concept_embedding: bool
