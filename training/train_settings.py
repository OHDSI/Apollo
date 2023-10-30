from configparser import ConfigParser
from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainingSettings:
    # system
    sequence_data_folder: str
    output_folder: str
    pretrained_model_folder: Optional[str]
    batch_size: int

    # learning objectives
    masked_concept_learning: bool
    mask_one_concept_per_visit: bool
    masked_visit_concept_learning: bool
    truncate_type: str
    label_prediction: bool

    # training
    train_fraction: float
    num_epochs: int
    num_freeze_epochs: int
    learning_rate: float
    weight_decay: float

    # model
    max_sequence_length: int
    hidden_size: int
    num_attention_heads: int
    num_hidden_layers: int
    intermediate_size: int
    hidden_act: str
    embedding_combination_method: str
    hidden_dropout_prob: float
    attention_probs_dropout_prob: float

    def __init__(self, config: ConfigParser):
        self.sequence_data_folder = config.get("system", "sequence_data_folder")
        self.output_folder = config.get("system", "output_folder")
        self.pretrained_model_folder = config.get("system", "pretrained_model_folder")
        if self.pretrained_model_folder.strip() == "":
            self.pretrained_model_folder = None
        self.batch_size = config.getint("system", "batch_size")

        self.masked_concept_learning = config.getboolean("learning objectives", "masked_concept_learning")
        self.mask_one_concept_per_visit = config.getboolean("learning objectives", "mask_one_concept_per_visit")
        self.masked_visit_concept_learning = config.getboolean("learning objectives", "masked_visit_concept_learning")
        self.truncate_type = config.get("learning objectives", "truncate_type")
        self.label_prediction = config.getboolean("learning objectives", "label_prediction")

        self.train_fraction = config.getfloat("training", "train_fraction")
        self.num_epochs = config.getint("training", "num_epochs")
        self.num_freeze_epochs = config.getint("training", "num_freeze_epochs")
        self.learning_rate = config.getfloat("training", "learning_rate")
        self.weight_decay = config.getfloat("training", "weight_decay")

        self.max_sequence_length = config.getint("model", "max_sequence_length")
        self.hidden_size = config.getint("model", "hidden_size")
        self.num_attention_heads = config.getint("model", "num_attention_heads")
        self.num_hidden_layers = config.getint("model", "num_hidden_layers")
        self.intermediate_size = config.getint("model", "intermediate_size")
        self.hidden_act = config.get("model", "hidden_act")
        self.embedding_combination_method = config.get("model", "embedding_combination_method")
        self.hidden_dropout_prob = config.getfloat("model", "hidden_dropout_prob")
        self.attention_probs_dropout_prob = config.getfloat("model", "attention_probs_dropout_prob")
        self._validate()

    def _validate(self):
        if self.truncate_type not in ["random", "tail"]:
            raise ValueError(f"Invalid truncate_type: {self.truncate_type}")
        if self.hidden_act not in ["relu", "gelu"]:
            raise ValueError(f"Invalid hidden_act: {self.hidden_act}")
        if self.embedding_combination_method not in ["concat", "sum"]:
            raise ValueError(f"Invalid embedding_combination_method: {self.embedding_combination_method}")

    def write_model_settings(self, filename: str) -> None:
        config = ConfigParser()
        config.add_section("model")
        config.set("model", "hidden_size", str(self.hidden_size))
        config.set("model", "num_attention_heads", str(self.num_attention_heads))
        config.set("model", "num_hidden_layers", str(self.num_hidden_layers))
        config.set("model", "intermediate_size", str(self.intermediate_size))
        config.set("model", "hidden_act", str(self.hidden_act))
        config.set("model", "embedding_combination_method", str(self.embedding_combination_method))
        config.set("model", "hidden_dropout_prob", str(self.hidden_dropout_prob))
        config.set("model", "attention_probs_dropout_prob", str(self.attention_probs_dropout_prob))
        with open(filename, "w") as config_file:
            config.write(config_file)
