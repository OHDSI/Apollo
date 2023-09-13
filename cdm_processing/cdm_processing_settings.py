from configparser import ConfigParser
from dataclasses import dataclass
from typing import List


@dataclass
class CdmProcesingSettings:
    cdm_data_path: str
    label_subfolder: str
    max_cores: int
    output_path: str
    map_drugs_to_ingredients: bool
    concepts_to_remove: List[int]
    profile: bool
    has_labels: bool

    def __init__(self, config: ConfigParser):
        self.cdm_data_path = config.get("system", "cdm_data_path")
        self.label_subfolder = config.get("system", "label_subfolder")
        self.max_cores = config.getint("system", "max_cores")
        self.output_path = config.get("system", "output_path")
        self.map_drugs_to_ingredients = config.getboolean("mapping", "map_drugs_to_ingredients")
        self.concepts_to_remove = [int(x) for x in config["mapping"].get("concepts_to_remove").split(",")]
        self.has_labels = config.getboolean("mapping", "has_labels")
        self.profile = config.getboolean("debug", "profile")

