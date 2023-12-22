from configparser import ConfigParser
from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class SimulationModelSettings:
    dynamic_state_count: int
    fixed_state_count: int
    concept_count: int
    serious_concept_count: int
    visit_multiplier: int
    days_to_simulate: int


@dataclass
class SimulationSettings:
    # system
    max_cores: int
    root_folder: str

    # pretraining data generation
    generate_pretraining_data: bool
    partition_count: int
    person_count: int

    # prediction data generation
    generate_prediction_data: bool
    prediction_concept_count: int
    partition_count: int
    train_person_count: int
    test_person_count: int
    prediction_window: int

    # simulation
    json_file_name: Optional[str]
    simulation_model_settings: Optional[SimulationModelSettings]

    # debug
    profile: bool
    log_verbosity: int

    def __init__(self, config: Dict[str, Any]):
        settings = config["system"]
        for key, value in settings.items():
            setattr(self, key, value)
        settings = config["pretraining data generation"]
        for key, value in settings.items():
            setattr(self, key, value)
        settings = config["prediction data generation"]
        for key, value in settings.items():
            setattr(self, key, value)
        settings: Dict[str, Any] = config["simulation"]
        self.json_file_name = settings["json_file_name"]
        if self.json_file_name is None:
            del(settings["json_file_name"])
            self.simulation_model_settings = SimulationModelSettings(**settings)
        settings = config["debug"]
        for key, value in settings.items():
            setattr(self, key, value)
        self.__post_init__()

    def __post_init__(self):
        if self.log_verbosity not in [0, 1, 2]:
            raise ValueError(f"Invalid log_verbosity: {self.log_verbosity}")
        if (self.simulation_model_settings is not None and
                self.prediction_concept_count > self.simulation_model_settings.concept_count):
            raise ValueError("The prediction_concept_count cannot be greater than concept_count")
