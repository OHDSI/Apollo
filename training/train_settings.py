from dataclasses import dataclass


@dataclass
class CehrBertSettings:
    sequence_data_folder: str
    output_folder: str
    batch_size: int = 32
    min_sequence_length: int = 5
    max_sequence_length: int = 512
    masked_language_model_learning_objective: bool = True
    visit_prediction_learning_objective: bool = False
    is_training: bool = True
    num_epochs: int = 10
    hidden_size: int = 768
    num_attention_heads: int = 12
    num_hidden_layers: int = 12
    intermediate_size: int = 3072
    hidden_act: str = "gelu"
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1