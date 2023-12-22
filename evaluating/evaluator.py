"""
Using a pre-trained model, fine-tune it on a new dataset and evaluate the performance on another dataset.
"""
import configparser
import logging
import os
import sys
from typing import List

import yaml

from cdm_processing.cdm_processing_settings import CdmProcessingSettings, MappingSettings
from cdm_processing.cdm_processor import CdmDataProcessor
from evaluating.evaluation_settings import EvaluationSettings
import utils.logger as logger
from training.train_model import ModelTrainer
from training.train_settings import ModelTrainingSettings, ModelSettings, TrainingSettings

LOGGER_FILE_NAME = "_evaluation_log.txt"


class Evaluator:

    def __init__(self, settings: EvaluationSettings):
        self._settings = settings
        os.makedirs(settings.output_folder, exist_ok=True)
        self._configure_logger()
        logger.log_settings(settings)
        self._cdm_mapping_config = configparser.ConfigParser()
        with open(os.path.join(settings.pretrained_model_folder, "cdm_mapping.yaml")) as file:
            self._mapping_settings = MappingSettings(**yaml.safe_load(file))
        with open(os.path.join(settings.pretrained_model_folder, "model.yaml")) as file:
            self._model_settings = ModelSettings(**yaml.safe_load(file))

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
            cdm_processing_settings = CdmProcessingSettings()
            cdm_processing_settings.mapping_settings = self._mapping_settings
            cdm_processing_settings.cdm_data_path = self._settings.train_data_folder
            cdm_processing_settings.label_sub_folder = self._settings.train_label_sub_folder
            cdm_processing_settings.output_path = sequence_data_folder
            cdm_processing_settings.max_cores = self._settings.max_cores
            cdm_processing_settings.profile = False
            cdm_data_processor = CdmDataProcessor(cdm_processing_settings)
            cdm_data_processor.process_cdm_data()

        if os.path.exists(self._settings.fine_tuned_model_folder):
            logging.info("Model folder '%s' already exists, skipping fine-tuning",
                         self._settings.fine_tuned_model_folder)
        else:
            logging.info("Fine-tune model using training data")
            model_training_settings = ModelTrainingSettings()
            model_training_settings.model_settings = self._model_settings
            model_training_settings.sequence_data_folder = sequence_data_folder
            model_training_settings.output_folder = self._settings.fine_tuned_model_folder
            model_training_settings.pretrained_model_folder = self._settings.pretrained_model_folder
            model_training_settings.batch_size = self._settings.batch_size
            model_training_settings.checkpoint_every = self._settings.checkpoint_every
            model_training_settings.training_settings = self._settings.training_settings
            model_training_settings.learning_objective_settings = self._settings.learning_objective_settings
            model_training_settings.learning_objective_settings.masked_concept_learning = False
            model_training_settings.learning_objective_settings.mask_one_concept_per_visit = False
            model_training_settings.learning_objective_settings.masked_visit_concept_learning = False
            model_trainer = ModelTrainer(settings=model_training_settings)
            model_trainer.train_model()

    def predict(self) -> None:
        sequence_data_folder = os.path.join(self._settings.test_data_folder,
                                            "person_sequence_" + self._settings.test_label_sub_folder)
        if os.path.exists(sequence_data_folder):
            logging.info("Sequence data folder '%s' already exists, skipping conversion", sequence_data_folder)
        else:
            logging.info("Convert CDM test data to sequence format")
            cdm_processing_settings = CdmProcessingSettings()
            cdm_processing_settings.mapping_settings = self._mapping_settings
            cdm_processing_settings.cdm_data_path = self._settings.test_data_folder
            cdm_processing_settings.label_sub_folder = self._settings.test_label_sub_folder
            cdm_processing_settings.output_path = sequence_data_folder
            cdm_processing_settings.max_cores = self._settings.max_cores
            cdm_processing_settings.profile = False
            cdm_data_processor = CdmDataProcessor(cdm_processing_settings)
            cdm_data_processor.process_cdm_data()

        evaluation_file = os.path.join(self._settings.output_folder,
                                       "evaluation" + self._settings.test_label_sub_folder + ".csv")
        if os.path.exists(evaluation_file):
            logging.info("Evaluation file '%s' already exists, skipping prediction", evaluation_file)
        else:
            logging.info("Predict on test data using trained model")
            model_training_settings = ModelTrainingSettings()
            model_training_settings.model_settings = self._model_settings
            model_training_settings.sequence_data_folder = sequence_data_folder
            model_training_settings.output_folder = self._settings.output_folder
            model_training_settings.pretrained_model_folder = self._settings.fine_tuned_model_folder
            model_training_settings.batch_size = self._settings.batch_size
            model_training_settings.checkpoint_every = self._settings.checkpoint_every
            training_settings = TrainingSettings(train_fraction=0,
                                                 num_epochs=1,
                                                 num_freeze_epochs=0,
                                                 learning_rate=self._settings.training_settings.learning_rate,
                                                 weight_decay=self._settings.training_settings.weight_decay)
            model_training_settings.training_settings = training_settings
            model_training_settings.learning_objective_settings = self._settings.learning_objective_settings
            model_training_settings.learning_objective_settings.masked_concept_learning = False
            model_training_settings.learning_objective_settings.mask_one_concept_per_visit = False
            model_training_settings.learning_objective_settings.masked_visit_concept_learning = False
            model_trainer = ModelTrainer(settings=model_training_settings)
            model_trainer.evaluate_model(evaluation_file)


def main(args: List[str]):
    with open(args[0]) as file:
        config = yaml.safe_load(file)
    training_settings = EvaluationSettings(config)
    evaluator = Evaluator(settings=training_settings)

    evaluator.evaluate()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise Exception("Must provide path to yaml file as argument")
    else:
        main(sys.argv[1:])
