from dataclasses import dataclass


@dataclass
class TrainingSettings:
    sequence_data_folder: str
    output_folder: str
    batch_size: int = 32
    min_sequence_length: int = 5
    max_sequence_length: int = 512
    masked_language_model_learning_objective: bool = True
    visit_prediction_learning_objective: bool = False
    do_evaluation: bool = True
    train_fraction: float = 0.8
    num_epochs: int = 10
    hidden_size: int = 768
    num_attention_heads: int = 8 #12
    num_hidden_layers: int = 5 #12
    intermediate_size: int = 768 # 3072
    hidden_act: str = "gelu"
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    learning_rate = 0.001
    weight_decay = 0.01
