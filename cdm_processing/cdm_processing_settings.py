from configparser import ConfigParser
from dataclasses import dataclass
from typing import List


@dataclass
class CdmProcessingSettings:
    cdm_data_path: str
    label_sub_folder: str
    max_cores: int
    output_path: str
    map_drugs_to_ingredients: bool
    concepts_to_remove: List[int]
    profile: bool
    has_labels: bool

    def __init__(self, config: ConfigParser):
        self.cdm_data_path = config.get("system", "cdm_data_path")
        self.label_sub_folder = config.get("system", "label_sub_folder")
        self.max_cores = config.getint("system", "max_cores")
        self.output_path = config.get("system", "output_path")
        self.map_drugs_to_ingredients = config.getboolean("mapping", "map_drugs_to_ingredients")
        self.concepts_to_remove = [int(x) for x in config["mapping"].get("concepts_to_remove").split(",")]
        self.has_labels = config.getboolean("mapping", "has_labels")
        self.profile = config.getboolean("debug", "profile")

    def write_mapping_settings(self, filename: str) -> None:
        config = ConfigParser()
        config.add_section("mapping")
        config.set("mapping", "map_drugs_to_ingredients", str(self.map_drugs_to_ingredients))
        config.set("mapping", "concepts_to_remove", ", ".join([str(x) for x in self.concepts_to_remove]))
        config.set("mapping", "has_labels", str(self.has_labels))
        with open(filename, "w") as config_file:
            config.write(config_file)
