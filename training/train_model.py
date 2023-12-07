"""
Train (or evaluate) a model on sequence data as generated by the CdmDataProcessor. This class can be used both for
training a model from scratch, and for fine-tuning a pre-trained model.
"""
import yaml
import logging
import os
import shutil
import sys
import time
from typing import List, Dict, Optional, Tuple

from model.simple_regression_model import SimpleRegressionModel

# Prevents error during validation when using torch 2.1.1:
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = str(1)
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import utils.logger as logger
from data_loading.model_inputs import InputTransformer

from training.train_settings import ModelTrainingSettings
from data_loading.dataset import ApolloDataset
from data_loading.data_transformer import ApolloDataTransformer
import data_loading.learning_objectives as learning_objectives
import data_loading.tokenizer as tokenizer
from data_loading.variable_names import DataNames
from model.model import TransformerModel
from utils.row import Row

LOGGER_FILE_NAME = "_model_training_log.txt"
BATCH_REPORT_INTERVAL = 1000
CONCEPT_TOKENIZER_FILE_NAME = "_concept_tokenizer.json"
VISIT_CONCEPT_TOKENIZER_FILE_NAME = "_visit_concept_tokenizer.json"


def _dict_to_device(data: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in data.items()}


def _get_file_name(epoch: int) -> str:
    return f"checkpoint_{epoch:03d}.pth"


def _find_latest_checkpoint(folder: str) -> Optional[str]:
    epoch = -1
    checkpoint = None
    with os.scandir(folder) as entries:
        for entry in entries:
            if entry.is_file() and entry.name.endswith(".pth"):
                new_epoch = int(entry.name[-7:-4])
                if new_epoch > epoch:
                    epoch = new_epoch
                    checkpoint = entry.path
    return checkpoint


class ModelTrainer:

    def __init__(self, settings: ModelTrainingSettings):
        self._settings = settings
        os.makedirs(settings.output_folder, exist_ok=True)
        self._configure_logger()
        logger.log_settings(settings)
        self._writer: Optional[SummaryWriter] = None

        # Get concept tokenizers:
        self._concept_tokenizer = self._get_concept_tokenizer(file_name=CONCEPT_TOKENIZER_FILE_NAME,
                                                              field_name=DataNames.CONCEPT_IDS)
        if settings.model_settings.visit_concept_embedding:
            self._visit_concept_tokenizer = self._get_concept_tokenizer(file_name=VISIT_CONCEPT_TOKENIZER_FILE_NAME,
                                                                        field_name=DataNames.VISIT_CONCEPT_IDS)

        # Get learning objectives:
        self._learning_objectives = []
        if settings.learning_objective_settings.masked_concept_learning:
            self._learning_objectives.append(learning_objectives.MaskedConceptLearningObjective(
                concept_tokenizer=self._concept_tokenizer,
                one_mask_per_visit=self._settings.learning_objective_settings.mask_one_concept_per_visit))
        if settings.learning_objective_settings.masked_visit_concept_learning:
            self._learning_objectives.append(learning_objectives.MaskedVisitConceptLearningObjective(
                visit_concept_tokenizer=self._visit_concept_tokenizer))
        if settings.learning_objective_settings.label_prediction:
            self._learning_objectives.append(learning_objectives.LabelPredictionLearningObjective())

        # Get data sets:
        self._train_data, self._test_data = self._get_data_sets()

        # Get model and optimizer:
        if torch.cuda.is_available():
            self._device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self._device = torch.device("mps")
        else:
            self._device = torch.device("cpu")
        if self._settings.model_settings.simple_regression_model:
            self._model = SimpleRegressionModel(model_settings=self._settings.model_settings,
                                                learning_objective_settings=self._settings.learning_objective_settings,
                                                tokenizer=self._concept_tokenizer,
                                                visit_tokenizer=self._visit_concept_tokenizer)
        else:
            self._model = TransformerModel(model_settings=self._settings.model_settings,
                                           learning_objective_settings=self._settings.learning_objective_settings,
                                           tokenizer=self._concept_tokenizer,
                                           visit_tokenizer=self._visit_concept_tokenizer)
        self._model.to(self._device)
        self._optimizer = optim.Adam(params=self._model.parameters(),
                                     lr=settings.training_settings.learning_rate,
                                     weight_decay=settings.training_settings.weight_decay)
        self._epoch = 0

    def _configure_logger(self) -> None:
        logger.create_logger(log_file_name=os.path.join(self._settings.output_folder, LOGGER_FILE_NAME))

    def _get_concept_tokenizer(self, file_name: str, field_name: str) -> tokenizer.ConceptTokenizer:
        if self._settings.pretrained_model_folder is not None:
            json_file = os.path.join(self._settings.pretrained_model_folder, file_name)
            logging.info("Loading pre-trained concept tokenizer for %s from %s", field_name, json_file)
            concept_tokenizer = tokenizer.load_from_json(json_file)
        else:
            json_file = os.path.join(self._settings.output_folder, file_name)
            if os.path.exists(json_file):
                logging.info("Loading concept tokenizer for %s from %s", field_name, json_file)
                concept_tokenizer = tokenizer.load_from_json(json_file)
            else:
                logging.info("Creating concept tokenizer for %s", field_name)
                concept_tokenizer = tokenizer.ConceptTokenizer()
                train_data = ApolloDataset(self._settings.sequence_data_folder, is_train=True)
                concept_tokenizer.fit_on_concept_sequences(train_data, field_name)
                concept_tokenizer.save_to_json(json_file)
        return concept_tokenizer

    def _get_data_sets(self) -> Tuple[Optional[ApolloDataset], Optional[ApolloDataset]]:
        input_transformer = InputTransformer(concept_tokenizer=self._concept_tokenizer,
                                             visit_tokenizer=self._visit_concept_tokenizer,
                                             model_settings=self._settings.model_settings)
        data_transformer = ApolloDataTransformer(learning_objectives=self._learning_objectives,
                                                 input_transformer=input_transformer,
                                                 max_sequence_length=self._settings.model_settings.max_sequence_length,
                                                 truncate_type=self._settings.learning_objective_settings.truncate_type)
        if self._settings.training_settings.train_fraction < 1.0:
            test_data = ApolloDataset(folder=self._settings.sequence_data_folder,
                                      data_transformer=data_transformer,
                                      train_test_split=self._settings.training_settings.train_fraction,
                                      is_train=False)
        else:
            test_data = None
        if self._settings.training_settings.train_fraction > 0.0:
            train_data = ApolloDataset(folder=self._settings.sequence_data_folder,
                                       data_transformer=data_transformer,
                                       train_test_split=self._settings.training_settings.train_fraction,
                                       is_train=True)
        else:
            train_data = None
        return train_data, test_data

    def _run_model(self, train: bool) -> None:
        """
        Run the model on the training or test data.
        Args:
            train: If true, run on the training data, otherwise run on the test data. When training, the model will be
                put in training mode, and the optimizer will be used to update the model parameters.
        """
        if train:
            self._model.train()
            dataset = self._train_data
        else:
            self._model.eval()
            dataset = self._test_data
        for learning_objective in self._learning_objectives:
            learning_objective.reset_performance_metrics()
        batch_count = 0
        start_time = time.time()
        data_loader = DataLoader(dataset=dataset,
                                 batch_size=self._settings.batch_size,
                                 num_workers=4,
                                 pin_memory=True)
        for inputs in data_loader:
            if (self._settings.training_settings.max_batches is not None and
                    batch_count >= self._settings.training_settings.max_batches):
                logging.info("Reached maximum number of batches specified by user, stopping")
                break
            batch_count += 1
            # for batch_count in range(1, 1000):

            inputs = _dict_to_device(inputs, self._device)
            if train:
                predictions = self._model(inputs)
            else:
                with torch.no_grad():
                    predictions = self._model(inputs)

            loss = 0.0
            first = True
            for learning_objective in self._learning_objectives:
                objective_loss = learning_objective.compute_loss(outputs=inputs,
                                                                 predictions=predictions)
                if first:
                    loss = objective_loss
                    first = False
                else:
                    loss += objective_loss

            if train:
                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()

                if batch_count % BATCH_REPORT_INTERVAL == 0:
                    elapsed = time.time() - start_time
                    logging.info("- Batches completed: %d, average time per batch: %s",
                                 batch_count,
                                 elapsed / batch_count)

        for learning_objective in self._learning_objectives:
            learning_objective.report_performance_metrics(train, self._writer, self._epoch)
        if self._writer is not None:
            self._writer.flush()

    def train_model(self) -> None:
        # Write the model,  cdm_mapping config files and tokenizers to the output folder so the trained model can be
        # used without the original config and tokenizer files:
        self._settings.write_model_settings(os.path.join(self._settings.output_folder, "model.yaml"))
        shutil.copy2(os.path.join(self._settings.sequence_data_folder, "cdm_mapping.yaml"),
                     self._settings.output_folder)
        if self._settings.pretrained_model_folder is not None:
            new_json_file = os.path.join(self._settings.output_folder, CONCEPT_TOKENIZER_FILE_NAME)
            self._concept_tokenizer.save_to_json(new_json_file)
            if (self._settings.learning_objective_settings.masked_visit_concept_learning or
                    self._settings.learning_objective_settings.label_prediction):
                new_json_file = os.path.join(self._settings.output_folder, VISIT_CONCEPT_TOKENIZER_FILE_NAME)
                self._visit_concept_tokenizer.save_to_json(new_json_file)

        logging.info("Performing computations on device: %s", self._device.type)
        logging.info("Total parameters: {:,} ".format(sum([param.nelement() for param in self._model.parameters()])))
        self._writer = SummaryWriter(self._settings.output_folder)
        self._load_checkpoint()
        start = self._epoch + 1
        if (self._settings.training_settings.num_freeze_epochs >= self._epoch and
                self._settings.pretrained_model_folder is not None):
            logging.info("Freezing pre-trained model weights")
            self._model.freeze_non_head()
        for self._epoch in range(start, self._settings.training_settings.num_epochs + 1):
            logging.info("Starting epoch %d", self._epoch)
            if self._train_data is not None:
                self._run_model(train=True)
                if (self._epoch == self._settings.training_settings.num_epochs or
                        (self._settings.checkpoint_every is not None and
                         self._epoch % self._settings.checkpoint_every == 0)):
                    self._save_checkpoint()
            if self._test_data is not None:
                self._run_model(train=False)
            if self._model.is_frozen() and self._epoch >= self._settings.training_settings.num_freeze_epochs:
                logging.info("Unfreezing pre-trained model weights")
                self._model.unfreeze_all()

    def evaluate_model(self, result_file: str, epoch: Optional[int] = None) -> None:
        if epoch is not None:
            self._load_model(file_name=os.path.join(self._settings.output_folder, _get_file_name(epoch)),
                             pretrained=True)
        else:
            self._load_checkpoint()
        self._run_model(train=False)
        row = Row()
        for learning_objective in self._learning_objectives:
            learning_objective.report_performance_metrics(train=False, writer=row, epoch=self._epoch)
        row.write_to_csv(result_file)

    def _save_checkpoint(self):
        file_name = os.path.join(self._settings.output_folder, _get_file_name(self._epoch))
        torch.save({
            "epoch": self._epoch,
            "model_state_dict": self._model.state_dict(),
            "optimizer_state_dict": self._optimizer.state_dict(),
        }, file_name)

    def _load_model(self, file_name: str, pretrained: bool = False) -> None:
        loaded = torch.load(file_name, map_location=self._device)
        self._model.load_state_dict(loaded["model_state_dict"], strict=not pretrained)
        if not pretrained:
            self._optimizer.load_state_dict(loaded["optimizer_state_dict"])
            self._epoch = loaded["epoch"]

    def _load_checkpoint(self) -> None:
        """
        Search for the latest checkpoint and load it. If no checkpoint is found, and a pre-trained model folder is
        specified, load the latest checkpoint from the pre-trained model folder.
        """
        file_name = _find_latest_checkpoint(self._settings.output_folder)
        if file_name is None:
            if (self._settings.pretrained_model_folder is not None and
                    not self._settings.model_settings.simple_regression_model):
                if self._settings.pretrained_epoch is not None:
                    file_name = os.path.join(self._settings.pretrained_model_folder,
                                             _get_file_name(self._settings.pretrained_epoch))
                else:
                    file_name = _find_latest_checkpoint(self._settings.pretrained_model_folder)
                logging.info("Loading pre-trained model from '%s'", file_name)
                self._load_model(file_name, pretrained=True)
            else:
                logging.info("No checkpoint found, starting with random weights")
        else:
            logging.info("Loading model from '%s'", file_name)
            self._load_model(file_name)


def main(args: List[str]):
    with open(args[0]) as file:
        config = yaml.safe_load(file)
    model_training_settings = ModelTrainingSettings(config)
    model_trainer = ModelTrainer(settings=model_training_settings)
    model_trainer.train_model()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise Exception("Must provide path to ini file as argument")
    else:
        main(sys.argv[1:])
