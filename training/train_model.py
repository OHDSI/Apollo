import configparser
import logging
import os
import sys
import time
from typing import List, Dict, Optional, Tuple

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import utils.logger as logger

from training.train_settings import TrainingSettings
from data_loading.dataset import ApolloDataset
from data_loading.data_transformer import ApolloDataTransformer
import data_loading.learning_objective as learning_objective
import data_loading.tokenizer as tokenizer
from data_loading.variable_names import DataNames, ModelInputNames, ModelOutputNames
from model.model import TransformerModel

# Inspired by https://pytorch.org/tutorials/beginner/transformer_tutorial.html

LOGGER_FILE_NAME = "_model_training_log.txt"
IGNORE_INDEX = -1
BATCH_REPORT_INTERVAL = 10


def _dict_to_device(data: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in data.items()}


def _masked_token_accuracy(token_predictions: torch.Tensor, token_ids: torch.Tensor) -> float:
    mask = (token_ids == IGNORE_INDEX)
    masked_predictions = token_predictions.argmax(2)[~mask]
    masked_labels = token_ids[~mask]
    return (masked_predictions == masked_labels).float().mean().item()


class ModelTrainer:

    def __init__(self,
                 settings: TrainingSettings):
        self._settings = settings
        os.makedirs(settings.output_folder, exist_ok=True)
        self._configure_logger()
        self._writer = SummaryWriter(settings.output_folder)
        self._concept_tokenizer = self._get_concept_tokenizer()
        self._visit_concept_tokenizer = self._get_visit_concept_tokenizer()
        self._train_data, self._test_data = self._get_data_sets()
        self._criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = TransformerModel(settings=self._settings,
                                       tokenizer=self._concept_tokenizer,
                                       visit_tokenizer=self._visit_concept_tokenizer)
        self._model.to(self._device)
        self._optimizer = optim.Adam(params=self._model.parameters(),
                                     lr=settings.learning_rate,
                                     weight_decay=settings.weight_decay)
        self._epoch = 0

    def _configure_logger(self) -> None:
        logger.create_logger(os.path.join(self._settings.output_folder, LOGGER_FILE_NAME))

    def _get_concept_tokenizer(self) -> tokenizer.ConceptTokenizer:
        json_file = os.path.join(self._settings.output_folder, "_concept_tokenizer.json")
        if os.path.exists(json_file):
            logging.info("Loading concept tokenizer from %s", json_file)
            concept_tokenizer = tokenizer.load_from_json(json_file)
        else:
            logging.info("Creating concept tokenizer")
            concept_tokenizer = tokenizer.ConceptTokenizer()
            train_data = ApolloDataset(self._settings.sequence_data_folder, is_train=True)
            concept_tokenizer.fit_on_concept_sequences(train_data, DataNames.CONCEPT_IDS)
            concept_tokenizer.save_to_json(json_file)
        return concept_tokenizer

    def _get_visit_concept_tokenizer(self) -> tokenizer.ConceptTokenizer:
        json_file = os.path.join(self._settings.output_folder, "_visit_concept_tokenizer.json")
        if os.path.exists(json_file):
            logging.info("Loading visit concept tokenizer from %s", json_file)
            visit_concept_tokenizer = tokenizer.load_from_json(json_file)
        else:
            logging.info("Creating visit concept tokenizer")
            visit_concept_tokenizer = tokenizer.ConceptTokenizer()
            train_data = ApolloDataset(self._settings.sequence_data_folder, is_train=True)
            visit_concept_tokenizer.fit_on_concept_sequences(train_data, DataNames.VISIT_CONCEPT_IDS)
            visit_concept_tokenizer.save_to_json(json_file)
        return visit_concept_tokenizer

    def _get_data_sets(self) -> Tuple[ApolloDataset, Optional[ApolloDataset]]:
        mlm_objective = learning_objective.MaskedLanguageModelLearningObjective(
            concept_tokenizer=self._concept_tokenizer,
            one_mask_per_visit=self._settings.masked_language_model_learning_objective_one_token_per_visit)
        visit_mlm_objective = learning_objective.VisitPredictionLearningObjective(
            visit_concept_tokenizer=self._visit_concept_tokenizer)
        learning_objectives = [mlm_objective, visit_mlm_objective]
        data_transformer = ApolloDataTransformer(learning_objectives=learning_objectives,
                                                 max_sequence_length=self._settings.max_sequence_length)
        if self._settings.do_evaluation:
            train_fraction = self._settings.train_fraction
            test_data = ApolloDataset(folder=self._settings.sequence_data_folder,
                                      data_transformer=data_transformer,
                                      is_train=False)
        else:
            train_fraction = 1.0
            test_data = None
        train_data = ApolloDataset(folder=self._settings.sequence_data_folder,
                                   data_transformer=data_transformer,
                                   train_test_split=train_fraction,
                                   is_train=True)
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
            print_label = "train"
        else:
            self._model.eval()
            dataset = self._test_data
            print_label = "tests"
            # Not applying no_grad() as a workaround for https://github.com/pytorch/pytorch/issues/97111
        total_token_loss = 0.
        total_token_accuracy = 0.
        total_visit_token_loss = 0.
        total_visit_token_accuracy = 0.
        batch_count = 0
        start_time = time.time()
        data_loader = DataLoader(dataset=dataset,
                                 batch_size=self._settings.batch_size,
                                 num_workers=4)
        for inputs, outputs in data_loader:
            batch_count += 1
            # for batch_count in range(1, 1000):

            inputs = _dict_to_device(inputs, self._device)
            outputs = _dict_to_device(outputs, self._device)
            predictions = self._model(inputs)
            token_predictions = predictions[ModelOutputNames.TOKEN_PREDICTIONS]
            visit_token_predictions = predictions[ModelOutputNames.VISIT_TOKEN_PREDICTIONS]

            # Compute token loss:
            token_ids = outputs[ModelInputNames.TOKEN_IDS]
            token_ids = token_ids.masked_fill(outputs[ModelInputNames.MASKED_TOKEN_MASK], IGNORE_INDEX)
            loss_token = self._criterion(token_predictions.transpose(1, 2), token_ids)
            total_token_loss += loss_token.float().mean().item()
            token_accuracy = _masked_token_accuracy(token_predictions, token_ids)
            total_token_accuracy += token_accuracy

            # Compute visit token loss:
            visit_token_ids = outputs[ModelInputNames.VISIT_TOKEN_IDS]
            visit_token_ids = visit_token_ids.masked_fill(outputs[ModelInputNames.MASKED_VISIT_TOKEN_MASK],
                                                          IGNORE_INDEX)
            loss_visit_token = self._criterion(visit_token_predictions.transpose(1, 2), visit_token_ids)
            total_visit_token_loss += loss_visit_token.float().mean().item()
            visit_token_accuracy = _masked_token_accuracy(visit_token_predictions, visit_token_ids)
            total_visit_token_accuracy += visit_token_accuracy

            if train:
                logging.info("Batch %d, Token loss: %0.2f, accuracy: %0.2f. Visit loss: %0.2f, accuracy: %0.2f",
                             batch_count,
                             loss_token.float().mean().item(),
                             token_accuracy,
                             loss_visit_token.float().mean().item(),
                             visit_token_accuracy)

                # Backpropagation:
                self._optimizer.zero_grad()
                loss = loss_token + loss_visit_token
                loss.backward()
                self._optimizer.step()

                if batch_count % BATCH_REPORT_INTERVAL == 0:
                    elapsed = time.time() - start_time
                    logging.info("Average time per batch: %s", elapsed / batch_count)

        logging.info("Mean token loss %s set: %0.2f, mean token accuracy %s set: %0.2f%%",
                     print_label,
                     total_token_loss / batch_count,
                     print_label,
                     100 * total_token_accuracy / batch_count)
        logging.info("Mean visit token loss %s set: %0.2f, mean token accuracy %s set: %0.2f%%",
                     print_label,
                     total_visit_token_loss / batch_count,
                     print_label,
                     100 * total_visit_token_accuracy / batch_count)
        self._writer.add_scalar(f"Mean token loss {print_label}",
                                total_token_loss / batch_count,
                                self._epoch)
        self._writer.add_scalar(f"Mean token accuracy {print_label}",
                                total_token_accuracy / batch_count,
                                self._epoch)
        self._writer.add_scalar(f"Mean visit token loss {print_label}",
                                total_visit_token_loss / batch_count,
                                self._epoch)
        self._writer.add_scalar(f"Mean visit token accuracy {print_label}",
                                total_visit_token_accuracy / batch_count,
                                self._epoch)

    def train_model(self) -> None:
        logging.info("Performing computations on device: %s", self._device.type)
        logging.info("Total parameters: {:,} ".format(sum([param.nelement() for param in self._model.parameters()])))
        self._load_checkpoint()
        start = self._epoch + 1
        for self._epoch in range(start, self._settings.num_epochs + 1):
            logging.info("Starting epoch %d", self._epoch)
            self._run_model(train=True)
            self._save_checkpoint()
            if self._settings.do_evaluation:
                self._run_model(train=False)

    def _save_checkpoint(self):
        file_name = os.path.join(self._settings.output_folder, f"checkpoint_{self._epoch:03d}.pth")
        torch.save({
            'epoch': self._epoch,
            'model_state_dict': self._model.state_dict(),
            'optimizer_state_dict': self._optimizer.state_dict(),
        }, file_name)

    def _load_checkpoint(self, epoch: Optional[int] = None) -> None:
        if epoch is None:
            epoch = 0
            with os.scandir(self._settings.output_folder) as entries:
                for entry in entries:
                    if entry.is_file() and entry.name.endswith(".pth"):
                        new_epoch = int(entry.name[-7:-4])
                        if new_epoch > epoch:
                            epoch = new_epoch
        if epoch == 0:
            logging.info("No checkpoint found, starting with clean model")
        else:
            file_name = os.path.join(self._settings.output_folder, f"checkpoint_{epoch:03d}.pth")
            logging.info("Loading model from '%s'", file_name)
            loaded = torch.load(file_name, map_location=self._device)
            self._epoch = loaded["epoch"]
            self._model.load_state_dict(loaded["model_state_dict"])
            self._optimizer.load_state_dict(loaded["optimizer_state_dict"])


def main(args: List[str]):
    config = configparser.ConfigParser()
    with open(args[0]) as file:  # Explicitly opening file so error is thrown when not found
        config.read_file(file)
    training_settings = TrainingSettings(
        sequence_data_folder=config.get("system", "sequence_data_folder"),
        output_folder=config.get("system", "output_folder"),
        batch_size=config.getint("data preparation", "batch_size"),
        min_sequence_length=config.getint("data preparation", "min_sequence_length"),
        max_sequence_length=config.getint("data preparation", "max_sequence_length"),
        masked_language_model_learning_objective=config.getboolean("data preparation",
                                                                   "masked_language_model_learning_objective"),
        visit_prediction_learning_objective=config.getboolean("data preparation",
                                                              "visit_prediction_learning_objective"),
        do_evaluation=config.getboolean("data preparation", "do_evaluation"),
        num_epochs=config.getint("training", "num_epochs"))
    model_trainer = ModelTrainer(settings=training_settings)
    # Log config after initializing model_trainer so logger is initialized:
    logger.log_config(config)
    model_trainer.train_model()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise Exception("Must provide path to ini file as argument")
    else:
        main(sys.argv[1:])
