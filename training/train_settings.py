import yaml
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any


@dataclass
class ModelSettings:
    max_sequence_length: int
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
class LearningObjectiveSettings:
    truncate_type: str
    label_prediction: bool
    masked_concept_learning: bool = False
    mask_one_concept_per_visit: bool = False
    masked_visit_concept_learning: bool = False

    def __post_init__(self):
        if self.truncate_type not in ["random", "tail"]:
            raise ValueError(f"Invalid truncate_type: {self.truncate_type}")


@dataclass
class TrainingSettings:
    train_fraction: float
    num_epochs: int
    num_freeze_epochs: int
    learning_rate: float
    weight_decay: float


@dataclass
class ModelTrainingSettings:
    # system
    sequence_data_folder: str
    output_folder: str
    pretrained_model_folder: Optional[str]
    batch_size: int
    training_settings: TrainingSettings
    learning_objective_settings: LearningObjectiveSettings
    model_settings: ModelSettings

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if config is None:
            return
        system = config["system"]
        for key, value in system.items():
            setattr(self, key, value)
        self.learning_objective_settings = LearningObjectiveSettings(**config["learning objectives"])
        self.training_settings = TrainingSettings(**config["training"])
        self.model_settings = ModelSettings(**config["model"])

    def write_model_settings(self, filename: str) -> None:
        with open(filename, "w") as config_file:
            yaml.dump(asdict(self.model_settings), config_file)
