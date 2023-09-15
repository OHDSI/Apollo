from configparser import ConfigParser
from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class TrainingSettings:
    # system
    sequence_data_folder: str
    output_folder: str
    pretrained_model_folder: Optional[str]

    # data preparation
    batch_size: int
    max_sequence_length: int
    truncate_type: str

    # learning objectives
    masked_concept_learning: bool
    mask_one_concept_per_visit: bool
    masked_visit_concept_learning: bool
    label_prediction: bool

    # training
    do_evaluation: bool
    train_fraction: float
    num_epochs: int
    num_freeze_epochs: int
    learning_rate: float
    weight_decay: float

    # model
    hidden_size: int
    num_attention_heads: int
    num_hidden_layers: int
    intermediate_size: int
    hidden_act: str
    hidden_dropout_prob: float
    attention_probs_dropout_prob: float

    def __init__(self, config: ConfigParser):
        self.sequence_data_folder = config.get("system", "sequence_data_folder")
        self.output_folder = config.get("system", "output_folder")
        self.pretrained_model_folder = config.get("system", "pretrained_model_folder")
        if self.pretrained_model_folder.strip() == "":
            self.pretrained_model_folder = None

        self.batch_size = config.getint("data preparation", "batch_size")
        self.max_sequence_length = config.getint("data preparation", "max_sequence_length")
        self.truncate_type = config.get("data preparation", "truncate_type")

        self.masked_concept_learning = config.getboolean("learning objectives", "masked_concept_learning")
        self.mask_one_concept_per_visit = config.getboolean("learning objectives", "mask_one_concept_per_visit")
        self.masked_visit_concept_learning = config.getboolean("learning objectives", "masked_visit_concept_learning")
        self.label_prediction = config.getboolean("learning objectives", "label_prediction")

        self.do_evaluation = config.getboolean("training", "do_evaluation")
        self.train_fraction = config.getfloat("training", "train_fraction")
        self.num_epochs = config.getint("training", "num_epochs")
        self.num_freeze_epochs = config.getint("training", "num_freeze_epochs")
        self.learning_rate = config.getfloat("training", "learning_rate")
        self.weight_decay = config.getfloat("training", "weight_decay")

        self.hidden_size = config.getint("model", "hidden_size")
        self.num_attention_heads = config.getint("model", "num_attention_heads")
        self.num_hidden_layers = config.getint("model", "num_hidden_layers")
        self.intermediate_size = config.getint("model", "intermediate_size")
        self.hidden_act = config.get("model", "hidden_act")
        self.hidden_dropout_prob = config.getfloat("model", "hidden_dropout_prob")
        self.attention_probs_dropout_prob = config.getfloat("model", "attention_probs_dropout_prob")
