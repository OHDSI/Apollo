import configparser
import logging
import os
import sys
from typing import List

import torch

import utils.logger as logger
import data_generating.data_generator as data_generator
import data_generating.learning_objective as learning_objective

LOGGER_FILE_NAME = "_model_training_log.txt"


class _CehrBertSettings:
    def __init__(self,
                 batch_size: int,
                 min_sequence_length: int,
                 max_sequence_length: int,
                 masked_language_model_learning_objective: bool,
                 visit_prediction_learning_objective: bool,
                 is_training: bool = True):
        self.batch_size = batch_size
        self.min_sequence_length = min_sequence_length
        self.max_sequence_length = max_sequence_length
        self.masked_language_model_learning_objective = masked_language_model_learning_objective
        self.visit_prediction_learning_objective = visit_prediction_learning_objective
        self.is_training = is_training


class CehrBertTrainer:

    def __init__(self,
                 sequence_data_folder: str,
                 output_folder: str,
                 cehr_bert_settings: _CehrBertSettings):
        self._sequence_data_folder = sequence_data_folder
        self._output_folder = output_folder
        os.makedirs(output_folder, exist_ok=True)
        self._configure_logger()
        self._cehr_bert_training_settings = cehr_bert_settings
        learning_objectives = []
        if cehr_bert_settings.masked_language_model_learning_objective:
            learning_objectives.append(learning_objective.MaskedLanguageModelLearningObjective(
                work_folder=output_folder,
                reuse_tokenizer=True))
        if cehr_bert_settings.visit_prediction_learning_objective:
            learning_objectives.append(learning_objective.VisitPredictionLearningObjective(
                work_folder=output_folder,
                reuse_tokenizer=True))
        self._data_generator = data_generator.DataGenerator(
            training_data_path=sequence_data_folder,
            batch_size=cehr_bert_settings.batch_size,
            min_sequence_length=cehr_bert_settings.min_sequence_length,
            max_sequence_length=cehr_bert_settings.max_sequence_length,
            is_training=cehr_bert_settings.is_training,
            learning_objectives=learning_objectives)

    def _configure_logger(self):
        logger.create_logger(os.path.join(self._output_folder, LOGGER_FILE_NAME))

    def train_model(self):
        logging.info("CUDA available: %s", torch.cuda.is_available())
        for batch in self._data_generator.generator():
            print("Input: masked_token_ids:")
            print(batch.get_inputs()["masked_token_ids"][0])
            print("Input: mask:")
            print(batch.get_inputs()["mask"][0])

            print("Output: token_ids:")
            print(batch.get_outputs()["token_ids"][0])
            print("Output: mask:")
            print(batch.get_outputs()["mask"][0])
            # Torch magic happens here
            break


def main(args: List[str]):
    config = configparser.ConfigParser()
    config.read(args[0])
    cehr_bert_settings = _CehrBertSettings(
        batch_size=config.getint("cehr-bert parameters", "batch_size"),
        min_sequence_length=config.getint("cehr-bert parameters", "min_sequence_length"),
        max_sequence_length=config.getint("cehr-bert parameters", "max_sequence_length"),
        masked_language_model_learning_objective=config.getboolean("cehr-bert parameters",
                                                                   "masked_language_model_learning_objective"),
        visit_prediction_learning_objective=config.getboolean("cehr-bert parameters",
                                                              "visit_prediction_learning_objective"),
        is_training=config.getboolean("cehr-bert parameters", "is_training"))
    cehr_bert_trainer = CehrBertTrainer(sequence_data_folder=config.get("system", "sequence_data_folder"),
                                        output_folder=config.get("system", "output_folder"),
                                        cehr_bert_settings=cehr_bert_settings)
    # Log config after initializing cehr_bert_trainer so logger is initialized:
    logger.log_config(config)
    cehr_bert_trainer.train_model()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise Exception("Must provide path to ini file as argument")
    else:
        main(sys.argv[1:])
