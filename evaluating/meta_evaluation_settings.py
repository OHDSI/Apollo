from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class MetaEvaluationSettings:
    # system
    root_folder: str
    sequence_data_folder: str
    train_data_folder: str
    train_label_sub_folders: List[str]
    test_data_folder: str
    test_label_sub_folders: List[str]
    max_cores: int
    batch_size: int
    pretrained_model_settings: List[Dict[str, Any]]
    fine_tuned_model_settings: List[Dict[str, Any]]

    def __init__(self, config: Dict[str, Any]):
        system = config["system"]
        for key, value in system.items():
            setattr(self, key, value)
        self.pretrained_model_settings = config["pretrained models"]
        self.fine_tuned_model_settings = config["fine-tuned models"]
