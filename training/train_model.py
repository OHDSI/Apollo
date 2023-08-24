import configparser
import logging
import os
import sys
import time
from typing import List

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
        self._model = TransformerModel(settings=self._settings, tokenizer=self._concept_tokenizer)
        self._print_model()
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

    def _get_data_sets(self) -> (ApolloDataset, ApolloDataset):
        mlm_objective = learning_objective.MaskedLanguageModelLearningObjective(self._concept_tokenizer)
        learning_objectives = [mlm_objective]
        data_transformer = ApolloDataTransformer(learning_objectives=learning_objectives,
                                                 max_sequence_length=self._settings.max_sequence_length)
        train_data = ApolloDataset(folder=self._settings.sequence_data_folder,
                                   data_transformer=data_transformer,
                                   is_train=True)
        test_data = ApolloDataset(folder=self._settings.sequence_data_folder,
                                  data_transformer=data_transformer,
                                  is_train=True)
        return train_data, test_data

    def _train(self) -> None:
        self._model.train()
        total_lml_loss = 0.
        batch_count = 0

        _data_loader = DataLoader(self._train_data, batch_size=self._settings.batch_size)
        for inputs, outputs in _data_loader:
            token_predictions = self._model(inputs)
            # This currently throws an error. May work on GPU:
            # if self._epoch == 1 and batch_count == 0:
            #     self._writer.add_graph(self._model, inputs)
            #     self._writer.flush()

            # Compute masked language model loss:
            token_ids = outputs[ModelInputNames.TOKEN_IDS]
            token_ids = token_ids.masked_fill(outputs[ModelInputNames.MASKED_TOKEN_MASK], IGNORE_INDEX)
            loss_token = self._criterion(token_predictions.transpose(1, 2), token_ids)

            loss = loss_token  # + loss_nsp
            batch_count += 1
            logging.info("Batch %d, Loss: %0.2f", batch_count, loss.tolist())
            total_lml_loss += loss.tolist()
            self._writer.add_scalar('training loss',
                                    loss,
                                    self._epoch * 1000 + batch_count)
            self._writer.flush()

            # Backpropagation:
            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()

        logging.info("Mean LML loss: %s", total_lml_loss / batch_count)
        print("Done")
        self._save_checkpoint()

    # def evaluate(model: nn.Module, eval_data: Tensor) -> float:
    #     model.eval()  # turn on evaluation mode
    #     total_loss = 0.
    #     with torch.no_grad():
    #         for i in range(0, eval_data.size(0) - 1, bptt):
    #             data, targets = get_batch(eval_data, i)
    #             seq_len = data.size(0)
    #             output = model(data)
    #             output_flat = output.view(-1, ntokens)
    #             total_loss += seq_len * criterion(output_flat, targets).item()
    #     return total_loss / (len(eval_data) - 1)

    def train_model(self) -> None:
        logging.info("CUDA available: %s", torch.cuda.is_available())

        for self._epoch in range(1, self._settings.num_epochs + 1):
            # epoch_start_time = time.time()
            self._train()
            # val_loss = evaluate(model, val_data)
            # val_ppl = math.exp(val_loss)
            # elapsed = time.time() - epoch_start_time
            # print('-' * 89)
            # print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
            #       f'valid loss {val_loss:5.2f} | valid ppl {val_ppl:8.2f}')
            # print('-' * 89)

            # if val_loss < best_val_loss:
            #     best_val_loss = val_loss
            #     torch.save(model.state_dict(), best_model_params_path)
            #
            # scheduler.step()
    def _save_checkpoint(self):
        file_name = os.path.join(self._settings.output_folder, f"checkpoint_{self._epoch}.pth")
        torch.save({
            'epoch': self._epoch,
            'model_state_dict': self._model.state_dict(),
            'optimizer_state_dict': self._optimizer.state_dict(),
            # 'loss': loss,
        }, file_name)

    def _print_model(self):
        print(self._model)
        total_params = sum([param.nelement() for param in self._model.parameters()])
        print("Total Params: {:,} ".format(total_params))


def main(args: List[str]):
    config = configparser.ConfigParser()
    with open(args[0]) as file:  # Explicitly opening file so error is thrown when not found
        config.read_file(file)
    cehr_bert_settings = TrainingSettings(
        sequence_data_folder=config.get("system", "sequence_data_folder"),
        output_folder=config.get("system", "output_folder"),
        batch_size=config.getint("data preparation", "batch_size"),
        min_sequence_length=config.getint("data preparation", "min_sequence_length"),
        max_sequence_length=config.getint("data preparation", "max_sequence_length"),
        masked_language_model_learning_objective=config.getboolean("data preparation",
                                                                   "masked_language_model_learning_objective"),
        visit_prediction_learning_objective=config.getboolean("data preparation",
                                                              "visit_prediction_learning_objective"),
        is_training=config.getboolean("data preparation", "is_training"))
    cehr_bert_trainer = ModelTrainer(settings=cehr_bert_settings)
    # Log config after initializing cehr_bert_trainer so logger is initialized:
    logger.log_config(config)
    cehr_bert_trainer.train_model()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise Exception("Must provide path to ini file as argument")
    else:
        main(sys.argv[1:])
