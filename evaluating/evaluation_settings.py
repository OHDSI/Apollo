from configparser import ConfigParser
from dataclasses import dataclass
from typing import Optional


@dataclass
class EvaluationSettings:
    # system
    pretrained_model_folder: str
    train_data_folder: str
    train_label_sub_folder: str
    test_data_folder: str
    test_label_sub_folder: str
    output_folder: str
    max_cores: int
    batch_size: int

    # learning objectives
    truncate_type: str
    label_prediction: bool

    # training
    do_evaluation: bool
    train_fraction: float
    num_epochs: int
    num_freeze_epochs: int
    learning_rate: float
    weight_decay: float

    def __init__(self, config: ConfigParser):
        self.pretrained_model_folder = config.get("system", "pretrained_model_folder")
        self.train_data_folder = config.get("system", "train_data_folder")
        self.train_label_sub_folder = config.get("system", "train_label_sub_folder")
        self.test_data_folder = config.get("system", "test_data_folder")
        self.test_label_sub_folder = config.get("system", "test_label_sub_folder")
        self.output_folder = config.get("system", "output_folder")
        self.max_cores = config.getint("system", "max_cores")
        self.batch_size = config.getint("system", "batch_size")

        self.truncate_type = config.get("learning objectives", "truncate_type")
        self.label_prediction = config.getboolean("learning objectives", "label_prediction")

        self.do_evaluation = config.getboolean("training", "do_evaluation")
        self.train_fraction = config.getfloat("training", "train_fraction")
        self.num_epochs = config.getint("training", "num_epochs")
        self.num_freeze_epochs = config.getint("training", "num_freeze_epochs")
        self.learning_rate = config.getfloat("training", "learning_rate")
        self.weight_decay = config.getfloat("training", "weight_decay")

        self._validate()

    def _validate(self):
        if self.truncate_type not in ["random", "tail"]:
            raise ValueError(f"Invalid truncate_type: {self.truncate_type}")
