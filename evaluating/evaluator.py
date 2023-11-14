import configparser
import logging
import os
import sys
from typing import List

from cdm_processing.cdm_processing_settings import CdmProcessingSettings
from cdm_processing.cdm_processor import CdmDataProcessor
from evaluating.evaluation_settings import EvaluationSettings
import utils.logger as logger
from training.train_model import ModelTrainer
from training.train_settings import TrainingSettings

LOGGER_FILE_NAME = "_evaluation_log.txt"


class Evaluator:

    def __init__(self,
                 settings: EvaluationSettings):
        self._settings = settings
        os.makedirs(settings.output_folder, exist_ok=True)
        self._configure_logger()
        logger.log_settings(settings)
        self._cdm_mapping_config = configparser.ConfigParser()
        with open(os.path.join(settings.pretrained_model_folder, "cdm_mapping.ini")) as file:
            self._cdm_mapping_config.read_file(file)
        self._model_config = configparser.ConfigParser()
        with open(os.path.join(settings.pretrained_model_folder, "model.ini")) as file:
            self._model_config.read_file(file)

    def _configure_logger(self) -> None:
        logger.create_logger(log_file_name=os.path.join(self._settings.output_folder, LOGGER_FILE_NAME))

    def evaluate(self) -> None:
        self.fine_tune()
        self.predict()

    def fine_tune(self) -> None:
        sequence_data_folder = os.path.join(self._settings.train_data_folder,
                                            "person_sequence_" + self._settings.train_label_sub_folder)
        if os.path.exists(sequence_data_folder):
            logging.info("Sequence data folder '%s' already exists, skipping conversion", sequence_data_folder)
        else:
            logging.info("Convert CDM training data to sequence format")
            cdm_mapping_config = configparser.ConfigParser()
            cdm_mapping_config.read_dict(self._cdm_mapping_config)
            cdm_mapping_config.add_section("system")
            cdm_mapping_config["system"]["cdm_data_path"] = self._settings.train_data_folder
            cdm_mapping_config["system"]["label_sub_folder"] = self._settings.train_label_sub_folder
            cdm_mapping_config["system"]["output_path"] = sequence_data_folder
            cdm_mapping_config["system"]["max_cores"] = str(self._settings.max_cores)
            cdm_mapping_config.add_section("debug")
            cdm_mapping_config["debug"]["profile"] = str(False)
            cdm_mapping_config["mapping"]["has_labels"] = str(True)
            cdm_processing_settings = CdmProcessingSettings(cdm_mapping_config)
            cdm_data_processor = CdmDataProcessor(cdm_processing_settings)
            cdm_data_processor.process_cdm_data()

        model_folder = os.path.join(self._settings.train_data_folder,
                                    "model_" + self._settings.train_label_sub_folder)
        if os.path.exists(model_folder):
            logging.info("Model folder '%s' already exists, skipping fine-tuning", model_folder)
        else:
            logging.info("Fine-tune model using training data")
            model_config = configparser.ConfigParser()
            model_config.read_dict(self._model_config)
            model_config.add_section("system")
            model_config["system"]["sequence_data_folder"] = sequence_data_folder
            model_config["system"]["output_folder"] = model_folder
            model_config["system"]["pretrained_model_folder"] = self._settings.pretrained_model_folder
            model_config["system"]["batch_size"] = str(self._settings.batch_size)
            model_config.add_section("learning objectives")
            model_config["learning objectives"]["truncate_type"] = self._settings.truncate_type
            model_config["learning objectives"]["label_prediction"] = str(self._settings.label_prediction)
            model_config["learning objectives"]["masked_concept_learning"] = str(False)
            model_config["learning objectives"]["mask_one_concept_per_visit"] = str(False)
            model_config["learning objectives"]["masked_visit_concept_learning"] = str(False)
            model_config.add_section("training")
            model_config["training"]["train_fraction"] = str(self._settings.train_fraction)
            model_config["training"]["num_epochs"] = str(self._settings.num_epochs)
            model_config["training"]["num_freeze_epochs"] = str(self._settings.num_freeze_epochs)
            model_config["training"]["learning_rate"] = str(self._settings.learning_rate)
            model_config["training"]["weight_decay"] = str(self._settings.weight_decay)
            training_settings = TrainingSettings(model_config)
            model_trainer = ModelTrainer(settings=training_settings)
            model_trainer.train_model()

    def predict(self) -> None:
        sequence_data_folder = os.path.join(self._settings.test_data_folder,
                                            "person_sequence_" + self._settings.test_label_sub_folder)
        if os.path.exists(sequence_data_folder):
            logging.info("Sequence data folder '%s' already exists, skipping conversion", sequence_data_folder)
        else:
            logging.info("Convert CDM test data to sequence format")
            cdm_mapping_config = configparser.ConfigParser()
            cdm_mapping_config.read_dict(self._cdm_mapping_config)
            cdm_mapping_config.add_section("system")
            cdm_mapping_config["system"]["cdm_data_path"] = self._settings.test_data_folder
            cdm_mapping_config["system"]["label_sub_folder"] = self._settings.test_label_sub_folder
            cdm_mapping_config["system"]["output_path"] = sequence_data_folder
            cdm_mapping_config["system"]["max_cores"] = str(self._settings.max_cores)
            cdm_mapping_config.add_section("debug")
            cdm_mapping_config["debug"]["profile"] = str(False)
            cdm_mapping_config["mapping"]["has_labels"] = str(True)
            cdm_processing_settings = CdmProcessingSettings(cdm_mapping_config)
            cdm_data_processor = CdmDataProcessor(cdm_processing_settings)
            cdm_data_processor.process_cdm_data()

        evaluation_file = os.path.join(self._settings.output_folder,
                                       "evaluation" + self._settings.test_label_sub_folder + ".csv")
        if os.path.exists(evaluation_file):
            logging.info("Evaluation file '%s' already exists, skipping prediction", evaluation_file)
        else:
            model_folder = os.path.join(self._settings.train_data_folder,
                                        "model_" + self._settings.train_label_sub_folder)
            logging.info("Predict on test data using trained model")
            model_config = configparser.ConfigParser()
            model_config.read_dict(self._model_config)
            model_config.add_section("system")
            model_config["system"]["sequence_data_folder"] = sequence_data_folder
            model_config["system"]["output_folder"] = self._settings.output_folder
            model_config["system"]["pretrained_model_folder"] = model_folder
            model_config["system"]["batch_size"] = str(self._settings.batch_size)
            model_config.add_section("learning objectives")
            model_config["learning objectives"]["truncate_type"] = self._settings.truncate_type
            model_config["learning objectives"]["label_prediction"] = str(self._settings.label_prediction)
            model_config["learning objectives"]["masked_concept_learning"] = str(False)
            model_config["learning objectives"]["mask_one_concept_per_visit"] = str(False)
            model_config["learning objectives"]["masked_visit_concept_learning"] = str(False)
            model_config.add_section("training")
            model_config["training"]["train_fraction"] = str(0)
            model_config["training"]["num_epochs"] = "1"
            model_config["training"]["num_freeze_epochs"] = "0"
            model_config["training"]["learning_rate"] = str(self._settings.learning_rate)
            model_config["training"]["weight_decay"] = str(self._settings.weight_decay)
            training_settings = TrainingSettings(model_config)
            model_trainer = ModelTrainer(settings=training_settings)
            model_trainer.evaluate_model(evaluation_file)


def main(args: List[str]):
    config = configparser.ConfigParser()
    with open(args[0]) as file:  # Explicitly opening file so error is thrown when not found
        config.read_file(file)
    training_settings = EvaluationSettings(config)
    evaluator = Evaluator(settings=training_settings)
    evaluator.evaluate()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise Exception("Must provide path to ini file as argument")
    else:
        main(sys.argv[1:])
