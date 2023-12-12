import logging
import random
from abc import abstractmethod
from typing import Dict, List, Union

import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.tensorboard import SummaryWriter
import sklearn.metrics as metrics

from data_loading.model_inputs import ModelInput, prefix_and_pad, find_last_index
from data_loading.tokenizer import ConceptTokenizer
from data_loading.variable_names import ModelInputNames, DataNames, ModelOutputNames
from utils.row import Row

IGNORE_INDEX = -1


class TokenPredictionPerformance:

    def __init__(self):
        self.sum_loss: float = 0
        self.sum_accuracy: float = 0
        self.n: int = 0

    def add(self, loss: float, accuracy: float) -> None:
        self.sum_loss += loss
        self.sum_accuracy += accuracy
        self.n += 1

    def reset(self) -> None:
        self.sum_loss = 0
        self.sum_accuracy = 0
        self.n = 0

    def get_mean_loss(self) -> float:
        return self.sum_loss / self.n

    def get_mean_accuracy(self) -> float:
        return self.sum_accuracy / self.n

    def report_metrics(self,
                       train: bool,
                       objective_label: str,
                       writer: [SummaryWriter, Row, None], epoch: int) -> None:
        label = "train" if train else "validation"
        label += " " + objective_label
        logging.info("Epoch %d %s mean loss: %0.2f, mean accuracy: %0.2f%%",
                     epoch,
                     label,
                     self.get_mean_loss(),
                     100 * self.get_mean_accuracy())
        if isinstance(writer, SummaryWriter):
            writer.add_scalar(f"{label} mean loss",
                              self.get_mean_loss(),
                              epoch)
            writer.add_scalar(f"{label} mean accuracy",
                              self.get_mean_accuracy(),
                              epoch)
        elif isinstance(writer, Row):
            writer.put_value(f"{label} mean loss", self.get_mean_loss())
            writer.put_value(f"{label} mean accuracy", self.get_mean_accuracy())


class BinaryPredictionPerformance:

    def __init__(self):
        self.sum_loss: float = 0
        self.predictions: list = []
        self.labels: list = []

    def add(self, loss: float, prediction: List[float], label: List[float]) -> None:
        self.sum_loss += loss
        self.predictions.extend(prediction)
        self.labels.extend(label)

    def reset(self) -> None:
        self.sum_loss = 0
        self.predictions = []
        self.labels = []

    def get_mean_loss(self) -> float:
        return self.sum_loss / len(self.predictions)

    def get_auc(self) -> float:
        fpr, tpr, thresholds = metrics.roc_curve(self.labels, self.predictions)
        return metrics.auc(fpr, tpr)

    def get_auprc(self) -> float:
        return metrics.average_precision_score(self.labels, self.predictions)

    def get_brier_score(self) -> float:
        return metrics.brier_score_loss(self.labels, self.predictions)

    def report_metrics(self,
                       train: bool,
                       objective_label: str,
                       writer: [SummaryWriter, Row, None],
                       epoch: int) -> None:
        label = "train" if train else "validation"
        label += " " + objective_label
        logging.info("Epoch %d %s mean loss: %0.2f, AUC: %0.2f, AUPRC: %0.2f, Brier score: %0.2f",
                     epoch,
                     label,
                     self.get_mean_loss(),
                     self.get_auc(),
                     self.get_auprc(),
                     self.get_brier_score())
        if isinstance(writer, SummaryWriter):
            writer.add_scalar(f"{label} mean loss",
                              self.get_mean_loss(),
                              epoch)
            writer.add_scalar(f"{label} AUC",
                              self.get_auc(),
                              epoch)
            writer.add_scalar(f"{label} AUPRC",
                              self.get_auprc(),
                              epoch)
            writer.add_scalar(f"{label} Brier score",
                              self.get_brier_score(),
                              epoch)
        elif isinstance(writer, Row):
            writer.put_value(f"{label} mean loss", self.get_mean_loss())
            writer.put_value(f"{label} AUC", self.get_auc())
            writer.put_value(f"{label} AUPRC", self.get_auprc())
            writer.put_value(f"{label} Brier score", self.get_brier_score())


def _masked_token_accuracy(token_predictions: torch.Tensor, token_ids: torch.Tensor) -> float:
    mask = (token_ids == IGNORE_INDEX)
    masked_predictions = token_predictions.argmax(2)[~mask]
    masked_labels = token_ids[~mask]
    return (masked_predictions == masked_labels).float().mean().item()


def _predicted_tokens_accuracy(tokens_predictions: torch.Tensor, visit_token_set: torch.Tensor) -> float:
    probabilities = torch.sigmoid(tokens_predictions)
    threshold = 0.5  # Probability threshold for a token to be considered predicted
    predictions = (probabilities > threshold)
    return (predictions == visit_token_set).float().mean().item()


class LearningObjective(ModelInput):
    """
    A learning objective is a task that can be learned from the data. For example, predicting the next visit. This
    class is used to generate the output data for the learning objective, as well as to compute the loss for the
    learning objective.
    """

    @abstractmethod
    def compute_loss(self, outputs: Dict[str, torch.Tensor], predictions: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute the loss for the learning objective, and update the performance metrics.
        Args:
            outputs: The expect outputs as generated by the process_row method. (Note that arrays are converted to
                tensors by the data generator.)
            predictions: The actual prediction of the model.

        Returns: The loss.
        """
        pass

    @abstractmethod
    def reset_performance_metrics(self) -> None:
        """
        Reset the performance metrics.
        """
        pass

    @abstractmethod
    def report_performance_metrics(self, train: bool, writer: [SummaryWriter, Row], epoch: int) -> None:
        """
        Report the performance metrics.
        Args:
            train: If true, the performance metrics are for the training set. Otherwise, they are for the validation
                set.
            writer: The tensorboard writer to use to write the performance metrics.
            epoch: The epoch number.
        """
        pass


class MaskedConceptLearningObjective(LearningObjective):

    def __init__(
            self,
            concept_tokenizer: ConceptTokenizer,
            one_mask_per_visit: bool = False
    ):
        """
        Initialization
        Args:
            concept_tokenizer: The tokenizer to use to tokenize the concepts. Should already be trained.
            one_mask_per_visit: If true, only one concept per visit is masked. Otherwise, multiple concepts per visit
                can be masked.
        """
        self._tokenizer = concept_tokenizer
        self._one_mask_per_visit = one_mask_per_visit
        self._criterion = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
        self._performance = TokenPredictionPerformance()

    def process_row(self,
                    row: Dict,
                    start_index: int,
                    end_index: int,
                    max_sequence_length: int) -> Dict[str, Union[NDArray, float]]:
        # Truncate the sequences:
        concept_ids = np.array(row[DataNames.CONCEPT_IDS][start_index:end_index])
        visit_concept_orders = np.array(row[DataNames.VISIT_CONCEPT_ORDERS][start_index:end_index])
        token_ids = self._tokenizer.encode(concept_ids)
        masked_token_ids, masked_token_mask = self._mask_tokens(token_ids, visit_concept_orders)
        masked_token_ids = prefix_and_pad(sequence=masked_token_ids,
                                          prefix_value=self._tokenizer.get_classification_token_id(),
                                          padding_value=self._tokenizer.get_padding_token_id(),
                                          max_sequence_length=max_sequence_length)
        masked_token_mask = prefix_and_pad(sequence=masked_token_mask,
                                           prefix_value=True,
                                           padding_value=True,
                                           max_sequence_length=max_sequence_length)
        model_inputs = {
            ModelInputNames.MASKED_TOKEN_IDS: masked_token_ids,
            ModelInputNames.MASKED_TOKEN_MASK: masked_token_mask
        }
        return model_inputs

    def _mask_tokens(self,
                     token_ids: NDArray[int],
                     visit_concept_orders: NDArray[int]
                     ) -> tuple[NDArray, NDArray]:
        masked_token_ids = token_ids.copy()
        masked_token_mask = np.ones(len(token_ids), dtype=bool)
        last_visit_order = 0
        for word_pos in range(0, len(token_ids)):
            if self._one_mask_per_visit and visit_concept_orders[word_pos] == last_visit_order:
                continue
            if random.random() < 0.15:
                dice = random.random()
                if dice < 0.8:
                    masked_token_ids[word_pos] = self._tokenizer.get_mask_token_id()
                elif dice < 0.9:
                    masked_token_ids[word_pos] = random.randint(
                        self._tokenizer.get_first_token_id(),
                        self._tokenizer.get_last_token_id())
                # else: 10% of the time we just leave the token as is
                masked_token_mask[word_pos] = False
                last_visit_order = visit_concept_orders[word_pos]
        return masked_token_ids, masked_token_mask

    def compute_loss(self, outputs: Dict[str, torch.Tensor], predictions: Dict[str, torch.Tensor]) -> torch.Tensor:
        token_predictions = predictions[ModelOutputNames.TOKEN_PREDICTIONS]
        token_ids = outputs[ModelInputNames.TOKEN_IDS]
        token_ids = token_ids.masked_fill(outputs[ModelInputNames.MASKED_TOKEN_MASK], IGNORE_INDEX)
        loss = self._criterion(token_predictions.transpose(1, 2), token_ids)
        token_accuracy = _masked_token_accuracy(token_predictions, token_ids)
        self._performance.add(loss=loss.float().mean().item(),
                              accuracy=token_accuracy)
        return loss

    def reset_performance_metrics(self) -> None:
        self._performance.reset()

    def report_performance_metrics(self, train: bool, writer: [SummaryWriter, Row], epoch: int) -> None:
        self._performance.report_metrics(train, "masked concept", writer, epoch)


class MaskedVisitConceptLearningObjective(LearningObjective):

    def __init__(self, visit_concept_tokenizer: ConceptTokenizer):
        """
        Initialization
        Args:
            visit_concept_tokenizer: The tokenizer to use to tokenize the visit concepts.
        """
        self._tokenizer = visit_concept_tokenizer
        self._criterion = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
        self._performance = TokenPredictionPerformance()

    def process_row(self,
                    row: Dict,
                    start_index: int,
                    end_index: int,
                    max_sequence_length: int) -> Dict[str, Union[NDArray, float]]:
        visit_concept_ids = np.array(row[DataNames.VISIT_CONCEPT_IDS][start_index:end_index])
        visit_token_ids = np.array(self._tokenizer.encode(visit_concept_ids))
        masked_visit_token_ids, masked_visit_token_mask = self._mask_visit_tokens(visit_token_ids)
        masked_visit_token_ids = prefix_and_pad(sequence=masked_visit_token_ids,
                                                prefix_value=self._tokenizer.get_classification_token_id(),
                                                padding_value=self._tokenizer.get_padding_token_id(),
                                                max_sequence_length=max_sequence_length)
        masked_visit_token_mask = prefix_and_pad(sequence=masked_visit_token_mask,
                                                 prefix_value=True,
                                                 padding_value=True,
                                                 max_sequence_length=max_sequence_length)

        model_inputs = {
            ModelInputNames.MASKED_VISIT_TOKEN_IDS: masked_visit_token_ids,
            ModelInputNames.MASKED_VISIT_TOKEN_MASK: masked_visit_token_mask
        }
        return model_inputs

    def _mask_visit_tokens(self,
                           visit_token_ids: NDArray[int]
                           ) -> tuple[NDArray, NDArray]:
        masked_visit_token_ids = np.asarray(visit_token_ids).copy()
        masked_visit_token_mask = np.ones(len(visit_token_ids), dtype=bool)
        for word_pos in range(0, len(visit_token_ids)):
            if random.random() < 0.5:
                masked_visit_token_mask[word_pos] = False
                masked_visit_token_ids[word_pos] = self._tokenizer.get_mask_token_id()
        return masked_visit_token_ids, masked_visit_token_mask

    def compute_loss(self, outputs: Dict[str, torch.Tensor], predictions: Dict[str, torch.Tensor]) -> torch.Tensor:
        visit_token_predictions = predictions[ModelOutputNames.VISIT_TOKEN_PREDICTIONS]
        visit_token_ids = outputs[ModelInputNames.VISIT_TOKEN_IDS]
        visit_token_ids = visit_token_ids.masked_fill(outputs[ModelInputNames.MASKED_VISIT_TOKEN_MASK],
                                                      IGNORE_INDEX)
        loss = self._criterion(visit_token_predictions.transpose(1, 2), visit_token_ids)
        visit_token_accuracy = _masked_token_accuracy(visit_token_predictions, visit_token_ids)
        self._performance.add(loss=loss.float().mean().item(),
                              accuracy=visit_token_accuracy)
        return loss

    def reset_performance_metrics(self) -> None:
        self._performance.reset()

    def report_performance_metrics(self, train: bool, writer: [SummaryWriter, Row], epoch: int) -> None:
        self._performance.report_metrics(train, "masked visit", writer, epoch)


class LabelPredictionLearningObjective(LearningObjective):

    def __init__(self):
        """
        Initialization
        """
        self._criterion = torch.nn.BCELoss()
        self._performance = BinaryPredictionPerformance()

    def process_row(self,
                    row: Dict,
                    start_index: int,
                    end_index: int,
                    max_sequence_length: int) -> Dict[str, Union[NDArray, float]]:
        label = np.float32(row[DataNames.LABEL])
        model_inputs = {
            ModelInputNames.FINETUNE_LABEL: label
        }
        return model_inputs

    def compute_loss(self, outputs: Dict[str, torch.Tensor], predictions: Dict[str, torch.Tensor]) -> torch.Tensor:
        label_predictions = predictions[ModelOutputNames.LABEL_PREDICTIONS]
        labels = outputs[ModelInputNames.FINETUNE_LABEL]

        loss = self._criterion(torch.squeeze(label_predictions), labels.float())
        self._performance.add(loss=loss.float().mean().item(),
                              prediction=label_predictions.detach().cpu().tolist(),
                              label=labels.detach().cpu().tolist())
        return loss

    def reset_performance_metrics(self) -> None:
        self._performance.reset()

    def report_performance_metrics(self, train: bool, writer: [SummaryWriter, Row], epoch: int) -> None:
        self._performance.report_metrics(train, "label prediction", writer, epoch)


class NextTokenLearningObjective(LearningObjective):

    def __init__(self, concept_tokenizer: ConceptTokenizer):
        """
        Initialization
        Args:
            concept_tokenizer: The tokenizer to use to tokenize the concepts.
        """
        self._tokenizer = concept_tokenizer
        self._criterion = torch.nn.CrossEntropyLoss()
        self._performance = TokenPredictionPerformance()

    def process_row(self,
                    row: Dict,
                    start_index: int,
                    end_index: int,
                    max_sequence_length: int) -> Dict[str, Union[NDArray, float]]:
        concept_ids = np.array(row[DataNames.CONCEPT_IDS][(start_index + 1):(end_index + 1)], dtype=str)
        token_ids = self._tokenizer.encode(concept_ids)
        token_ids = prefix_and_pad(sequence=token_ids,
                                   prefix_value=self._tokenizer.get_classification_token_id(),
                                   padding_value=self._tokenizer.get_padding_token_id(),
                                   max_sequence_length=max_sequence_length)
        model_inputs = {
            ModelInputNames.NEXT_TOKEN_IDS: token_ids,
        }
        return model_inputs

    def compute_loss(self, outputs: Dict[str, torch.Tensor], predictions: Dict[str, torch.Tensor]) -> torch.Tensor:
        next_token_prediction = predictions[ModelOutputNames.NEXT_TOKEN_PREDICTION]
        next_token_ids = outputs[ModelInputNames.NEXT_TOKEN_IDS]
        loss = self._criterion(next_token_prediction.transpose(1, 2), next_token_ids)
        token_accuracy = _masked_token_accuracy(next_token_prediction, next_token_ids)
        self._performance.add(loss=loss.float().mean().item(),
                              accuracy=token_accuracy)
        return loss

    def reset_performance_metrics(self) -> None:
        self._performance.reset()

    def report_performance_metrics(self, train: bool, writer: [SummaryWriter, Row], epoch: int) -> None:
        self._performance.report_metrics(train, "next token", writer, epoch)


class NextVisitConceptsLearningObjective(LearningObjective):

    def __init__(self, concept_tokenizer: ConceptTokenizer):
        """
        Initialization
        Args:
            concept_tokenizer: The tokenizer to use to tokenize the concepts.
        """
        self._tokenizer = concept_tokenizer
        self._criterion = torch.nn.BCEWithLogitsLoss()
        self._performance = TokenPredictionPerformance()

    def process_row(self,
                    row: Dict,
                    start_index: int,
                    end_index: int,
                    max_sequence_length: int) -> Dict[str, Union[NDArray, float]]:
        start_next_visit_index = end_index + 1
        next_visit = row[DataNames.VISIT_CONCEPT_ORDERS][start_next_visit_index]
        end_next_visit_index = find_last_index(row[DataNames.VISIT_CONCEPT_ORDERS], next_visit) + 1
        concept_ids = np.array(row[DataNames.CONCEPT_IDS][start_next_visit_index:end_next_visit_index], dtype=str)
        token_ids = self._tokenizer.encode(concept_ids)
        next_visit_token_set = np.zeros(self._tokenizer.get_vocab_size(), dtype=np.float32)
        next_visit_token_set[token_ids] = 1
        model_inputs = {
            ModelInputNames.NEXT_VISIT_TOKEN_SET: next_visit_token_set,
        }
        return model_inputs

    def compute_loss(self, outputs: Dict[str, torch.Tensor], predictions: Dict[str, torch.Tensor]) -> torch.Tensor:
        next_visit_tokens_prediction = predictions[ModelOutputNames.NEXT_VISIT_TOKENS_PREDICTION]
        next_visit_token_set = outputs[ModelInputNames.NEXT_VISIT_TOKEN_SET]
        loss = self._criterion(next_visit_tokens_prediction, next_visit_token_set)
        token_accuracy = +_predicted_tokens_accuracy(next_visit_tokens_prediction, next_visit_token_set)
        self._performance.add(loss=loss.float().mean().item(),
                              accuracy=token_accuracy)
        return loss

    def reset_performance_metrics(self) -> None:
        self._performance.reset()

    def report_performance_metrics(self, train: bool, writer: [SummaryWriter, Row], epoch: int) -> None:
        self._performance.report_metrics(train, "next visit tokens", writer, epoch)
