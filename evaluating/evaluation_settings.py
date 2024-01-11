from dataclasses import dataclass
from typing import Dict, Any, Optional

from training.train_settings import LearningObjectiveSettings, TrainingSettings


@dataclass
class EvaluationSettings:
    # system
    pretrained_model_folder: str
    fine_tuned_model_folder: str
    train_data_folder: str
    train_label_sub_folder: str
    test_data_folder: str
    test_label_sub_folder: str
    output_folder: str
    max_cores: int
    batch_size: int
    checkpoint_every: Optional[int]
    learning_objective_settings: LearningObjectiveSettings
    training_settings: TrainingSettings

    def __init__(self, config: Dict[str, Any]):
        system = config["system"]
        for key, value in system.items():
            setattr(self, key, value)
        self.learning_objective_settings = LearningObjectiveSettings(**config["learning_objectives"])
        self.training_settings = TrainingSettings(**config["training"])
