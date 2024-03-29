import yaml
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any

from model.model_settings import ModelSettings


@dataclass
class LearningObjectiveSettings:
    truncate_type: str
    label_prediction: bool = False
    masked_concept_learning: bool = False
    mask_one_concept_per_visit: bool = True
    masked_visit_concept_learning: bool = False
    next_token_prediction: bool = False
    next_visit_concepts_prediction: bool = False
    simple_regression_model: bool = False

    def __post_init__(self):
        if self.truncate_type not in ["random", "tail"]:
            raise ValueError(f"Invalid truncate_type: {self.truncate_type}")


@dataclass
class TrainingSettings:
    train_fraction: float
    num_epochs: int
    learning_rate: float
    weight_decay: float
    num_freeze_epochs: int = 0
    max_batches: Optional[int] = None


@dataclass
class ModelTrainingSettings:
    # system
    sequence_data_folder: str
    output_folder: str
    pretrained_model_folder: Optional[str]
    batch_size: int
    checkpoint_every: Optional[int]
    training_settings: TrainingSettings
    learning_objective_settings: LearningObjectiveSettings
    model_settings: ModelSettings
    pretrained_epoch: Optional[int] = None

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if config is None:
            return
        system = config["system"]
        for key, value in system.items():
            setattr(self, key, value)
        self.learning_objective_settings = LearningObjectiveSettings(**config["learning objectives"])
        self.training_settings = TrainingSettings(**config["training"])
        self.model_settings = ModelSettings(**config["model"])

    def __post_init__(self):
        if self.learning_objective_settings.masked_concept_learning and not self.model_settings.concept_embedding:
            raise ValueError("Must have concept embedding if masked_concept_learning is true")
        if (self.learning_objective_settings.masked_visit_concept_learning and
                not self.model_settings.visit_concept_embedding):
            raise ValueError("Must have visit concept embedding if masked_visit_concept_learning is true")
        if self.learning_objective_settings.next_token_prediction and not self.model_settings.concept_embedding:
            raise ValueError("Must have concept embedding if next_token_prediction is true")
        if self.learning_objective_settings.simple_regression_model:
            if self.learning_objective_settings.label_prediction:
                raise ValueError("Must have label prediction if simple_regression_model is true")
            if (self.learning_objective_settings.masked_concept_learning or
                    self.learning_objective_settings.masked_visit_concept_learning):
                raise ValueError("Masked concept learning and masked visit concept learning are not implemented "
                                 "for the simple regression model.")
        if self.learning_objective_settings.next_token_prediction:
            if (self.learning_objective_settings.label_prediction or
                    self.learning_objective_settings.masked_concept_learning or
                    self.learning_objective_settings.masked_visit_concept_learning or
                    self.learning_objective_settings.next_visit_concepts_prediction):
                raise ValueError("Cannot combine next token prediction with any of the other learning objectives.")

    def write_model_settings(self, filename: str) -> None:
        with open(filename, "w") as config_file:
            yaml.dump(asdict(self.model_settings), config_file)
