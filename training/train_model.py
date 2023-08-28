import configparser
import logging
import os
import sys
import time
from typing import List, Dict, Optional

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
from data_loading.variable_names import DataNames, ModelInputNames
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
        self._train_data, self._test_data = self._get_data_sets()
        self._criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = TransformerModel(settings=self._settings, tokenizer=self._concept_tokenizer)

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

    def _get_data_sets(self) -> (ApolloDataset, Optional[ApolloDataset]):
        mlm_objective = learning_objective.MaskedLanguageModelLearningObjective(self._concept_tokenizer)
        learning_objectives = [mlm_objective]
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

    def _train(self) -> None:
        self._model.train()
        total_lml_loss = 0.
        total_accuracy = 0.
        batch_count = 0
        epoch_start_time = time.time()
        start_time = epoch_start_time
        _data_loader = DataLoader(dataset=self._train_data,
                                  batch_size=self._settings.batch_size,
                                  num_workers=4)
        for inputs, outputs in _data_loader:
            batch_count += 1
            # for batch_count in range(1000):

            inputs = _dict_to_device(inputs, self._device)
            outputs = _dict_to_device(outputs, self._device)
            token_predictions = self._model(inputs)

            # Compute masked language model loss:
            token_ids = outputs[ModelInputNames.TOKEN_IDS]
            token_ids = token_ids.masked_fill(outputs[ModelInputNames.MASKED_TOKEN_MASK], IGNORE_INDEX)
            loss_token = self._criterion(token_predictions.transpose(1, 2), token_ids)

            loss = loss_token  # + loss_nsp
            total_lml_loss += loss.float().mean().item()
            logging.info("Batch %d, Loss: %0.2f", batch_count, loss.tolist())
            total_accuracy += _masked_token_accuracy(token_predictions, token_ids)

            self._writer.add_scalar('Per-batch training loss',
                                    loss.float().mean(),
                                    self._epoch * 1000 + batch_count)
            self._writer.flush()

            # Backpropagation:
            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()

            if batch_count % BATCH_REPORT_INTERVAL == 0:
                elapsed = time.time() - start_time
                logging.info("Elapsed time: %s", elapsed)
                start_time = time.time()
                # for i in range(self._settings.max_sequence_length):
                #     if token_ids[0, i] != IGNORE_INDEX:
                #         print(token_predictions[0, i, token_ids[0, i]])

        logging.info("Mean LML loss train set: %0.2f, mean LML accuracy train set: %0.2f%%",
                     total_lml_loss / batch_count,
                     100 * total_accuracy / batch_count)
        self._writer.add_scalar('Mean LML loss train set',
                                total_lml_loss / batch_count,
                                self._epoch)
        self._writer.add_scalar('Mean LML accuracy train'
                                ' set',
                                total_accuracy / batch_count,
                                self._epoch)


    def _evaluate(self) -> None:
        self._model.eval()  # turn on evaluation mode
        total_lml_loss = 0.
        total_accuracy = 0.
        batch_count = 0
        _data_loader = DataLoader(dataset=self._test_data,
                                  batch_size=self._settings.batch_size,
                                  num_workers=4)
        # Not applying no_grad() as a workaround for https://github.com/pytorch/pytorch/issues/97111
        # with torch.no_grad():
        for inputs, outputs in _data_loader:
            batch_count += 1

            inputs = _dict_to_device(inputs, self._device)
            outputs = _dict_to_device(outputs, self._device)
            token_predictions = self._model(inputs)

            # Compute masked language model loss:
            token_ids = outputs[ModelInputNames.TOKEN_IDS]
            token_ids = token_ids.masked_fill(outputs[ModelInputNames.MASKED_TOKEN_MASK], IGNORE_INDEX)
            loss_token = self._criterion(token_predictions.transpose(1, 2), token_ids)

            loss = loss_token  # + loss_nsp
            total_lml_loss += loss.tolist()

            total_accuracy += _masked_token_accuracy(token_predictions, token_ids)

        logging.info("Mean LML loss test set: %0.2f, mean LML accuracy test set: %0.2f%%",
                     total_lml_loss / batch_count,
                     100 * total_accuracy / batch_count)
        self._writer.add_scalar('Mean LML loss test set',
                                total_lml_loss / batch_count,
                                self._epoch)
        self._writer.add_scalar('Mean LML accuracy test set',
                                total_accuracy / batch_count,
                                self._epoch)

    def train_model(self) -> None:
        logging.info("Performing computations on device: %s", self._device.type)
        logging.info("Total parameters: {:,} ".format(sum([param.nelement() for param in self._model.parameters()])))
        self._load_checkpoint()
        start = self._epoch + 1
        for self._epoch in range(start, self._settings.num_epochs + 1):
            logging.info("Starting epoch %d", self._epoch)
            self._train()
            self._save_checkpoint()
            if self._settings.do_evaluation:
                self._evaluate()

    def _save_checkpoint(self):
        file_name = os.path.join(self._settings.output_folder, f"checkpoint_{self._epoch:03d}.pth")
        torch.save({
            'epoch': self._epoch,
            'model_state_dict': self._model.state_dict(),
            'optimizer_state_dict': self._optimizer.state_dict(),
            # 'loss': loss,
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
