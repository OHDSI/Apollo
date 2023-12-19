from configparser import ConfigParser
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional

import yaml


@dataclass
class MappingSettings:
    map_drugs_to_ingredients: bool
    concepts_to_remove: List[int]


@dataclass
class CdmProcessingSettings:
    cdm_data_path: str
    max_cores: int
    output_path: str
    mapping_settings: MappingSettings
    profile: bool
    label_sub_folder: Optional[str] = None

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if config is None:
            return
        system = config["system"]
        for key, value in system.items():
            setattr(self, key, value)
        self.mapping_settings = MappingSettings(**config["mapping"])
        debug = config["debug"]
        for key, value in debug.items():
            setattr(self, key, value)

    def write_mapping_settings(self, filename: str) -> None:
        with open(filename, "w") as config_file:
            yaml.dump(asdict(self.mapping_settings), config_file)
