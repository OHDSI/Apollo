"""
Run evaluations on different models, for different concepts, using different settings.
"""

import logging
import os
import sys
from typing import List, Dict

import pandas as pd
import yaml

from evaluating.evaluation_settings import EvaluationSettings
from evaluating.evaluator import Evaluator
from evaluating.meta_evaluation_settings import MetaEvaluationSettings
from training.train_model import ModelTrainer
from training.train_settings import ModelTrainingSettings
from utils import logger

LOGGER_FILE_NAME = "_meta_evaluation_log.txt"


def _to_file_name(string: str) -> str:
    return "".join(x for x in string if x.isalnum())


class MetaEvaluator:

    def __init__(self, settings: MetaEvaluationSettings):
        self._settings = settings
        os.makedirs(settings.root_folder, exist_ok=True)
        self._configure_logger()
        logger.log_settings(settings)
        self._pretrained_model_settings = self._parse_pretrained_model_settings()
        self._evaluation_settings = self._parse_fine_tuned_model_settings()

    def _configure_logger(self) -> None:
        logger.create_logger(log_file_name=os.path.join(self._settings.root_folder, LOGGER_FILE_NAME))

    def _parse_pretrained_model_settings(self) -> Dict[str, ModelTrainingSettings]:
        pretrained_model_settings = {}
        for settings in self._settings.pretrained_model_settings:
            system = {"output_folder": os.path.join(self._settings.root_folder,
                                                    "pretrained_" + _to_file_name(settings["name"])),
                      "sequence_data_folder": self._settings.sequence_data_folder,
                      "max_cores": self._settings.max_cores,
                      "batch_size": self._settings.batch_size,
                      "pretrained_model_folder": None}
            settings["system"] = system
            pretrained_model_settings[settings["name"]] = ModelTrainingSettings(settings)
        logging.info("Number of pretrained models: %d", len(pretrained_model_settings))
        return pretrained_model_settings

    def _parse_fine_tuned_model_settings(self):
        evaluation_settings = []
        for settings in self._settings.fine_tuned_model_settings:
            for pretrained_model in settings["pretrained models"]:
                for i in range(len(self._settings.train_label_sub_folders)):
                    pretrained_model_name = pretrained_model["name"]
                    pretrained_mode_settings = self._pretrained_model_settings[pretrained_model_name]
                    pretrained_epoch = pretrained_model.get("pretrained_epoch", None)
                    if pretrained_epoch is None:
                        pretrained_epoch = pretrained_mode_settings.training_settings.num_epochs
                    else:
                        if pretrained_model["pretrained_epoch"] > pretrained_mode_settings.training_settings.num_epochs:
                            raise ValueError("pretrained_epoch must be <= num_epochs")
                    eval_folder = os.path.join(self._settings.root_folder,
                                               "eval_" +
                                               _to_file_name(pretrained_model["name"]) +
                                               "_e" +
                                               str(pretrained_epoch) +
                                               "_" +
                                               _to_file_name(settings["name"]))
                    fine_tuned_model_folder = os.path.join(self._settings.root_folder,
                                                           "model_" +
                                                           _to_file_name(pretrained_model["name"]) +
                                                           "_e" +
                                                           str(pretrained_epoch) +
                                                           "_" +
                                                           _to_file_name(settings["name"]) +
                                                           "_" +
                                                           _to_file_name(self._settings.train_label_sub_folders[i]))
                    system = {"pretrained_model_folder": os.path.join(self._settings.root_folder,
                                                                      "pretrained_" +
                                                                      _to_file_name(pretrained_model["name"])),
                              "fine_tuned_model_folder": fine_tuned_model_folder,
                              "train_data_folder": self._settings.train_data_folder,
                              "train_label_sub_folder": self._settings.train_label_sub_folders[i],
                              "test_data_folder": self._settings.test_data_folder,
                              "test_label_sub_folder": self._settings.test_label_sub_folders[i],
                              "output_folder": eval_folder,
                              "max_cores": self._settings.max_cores,
                              "batch_size": self._settings.batch_size,
                              "pretrained_epoch": pretrained_epoch}
                    settings["system"] = system
                    evaluation_settings.append(EvaluationSettings(settings))
        logging.info("Number of fine-tuned models: %d", len(evaluation_settings))
        return evaluation_settings

    def evaluate(self) -> None:
        for pretrained_model, settings in self._pretrained_model_settings.items():
            if os.path.exists(settings.output_folder):
                logging.info("Pretrained model '%s' already exists, skipping training", pretrained_model)
                continue
            logging.info("Train model '%s'", pretrained_model)
            trainer = ModelTrainer(self._pretrained_model_settings[pretrained_model])
            trainer.train_model()

        for evaluation_setting in self._evaluation_settings:
            logging.info("Running evaluation '%s'", evaluation_setting.output_folder)
            evaluator = Evaluator(evaluation_setting)
            evaluator.evaluate()

        self._combine_results()

    def _combine_results(self) -> None:
        rows = []
        for settings in self._settings.fine_tuned_model_settings:
            for pretrained_model in settings["pretrained models"]:
                for i in range(len(self._settings.train_label_sub_folders)):
                    pretrained_model_name = pretrained_model["name"]
                    pretrained_mode_settings = self._pretrained_model_settings[pretrained_model_name]
                    pretrained_epoch = pretrained_model.get("pretrained_epoch", None)
                    if pretrained_epoch is None:
                        pretrained_epoch = pretrained_mode_settings.training_settings.num_epochs
                    else:
                        if pretrained_model["pretrained_epoch"] > pretrained_mode_settings.training_settings.num_epochs:
                            raise ValueError("pretrained_epoch must be <= num_epochs")
                    eval_file = os.path.join(self._settings.root_folder,
                                             "eval_" +
                                             _to_file_name(pretrained_model) +
                                             "_" +
                                             str(pretrained_epoch) +
                                             "_" +
                                             _to_file_name(settings["name"]),
                                             "evaluation" +
                                             self._settings.test_label_sub_folders[i] +
                                             ".csv")
                    row = pd.read_csv(eval_file)
                    row["pretrained_model"] = pretrained_model
                    row["pretrained_epoch"] = pretrained_epoch
                    row["fine_tuned_model"] = settings["name"]
                    row["test_label_sub_folders"] = self._settings.test_label_sub_folders[i]
                    rows.append(row)
        df = pd.concat(rows)
        df.to_csv(os.path.join(self._settings.root_folder, "evaluation.csv"), index=False)

        file_name = os.path.join(self._settings.root_folder, "evaluation_pretrained_fine_tuned.csv")
        (df.drop(columns="test_label_sub_folders")
           .groupby(["pretrained_model", "fine_tuned_model"])
           .mean()
           .to_csv(file_name))

        file_name = os.path.join(self._settings.root_folder, "evaluation_pretrained.csv")
        (df.drop(columns=["test_label_sub_folders", "fine_tuned_model"])
         .groupby(["pretrained_model"])
         .mean()
         .to_csv(file_name))


def main(args: List[str]):
    with open(args[0]) as file:
        config = yaml.safe_load(file)
    meta_evaluation_settings = MetaEvaluationSettings(config)
    evaluator = MetaEvaluator(settings=meta_evaluation_settings)
    evaluator.evaluate()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise Exception("Must provide path to ini file as argument")
    else:
        main(sys.argv[1:])
