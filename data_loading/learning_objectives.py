import logging
import random
from abc import ABC, abstractmethod
from typing import Dict

import numpy as np
import torch
from accelerate import Accelerator

from data_loading.tokenizer import ConceptTokenizer
from data_loading.variable_names import ModelInputNames, DataNames, ModelOutputNames


IGNORE_INDEX = -1


def _prefix_and_pad(sequence: np.ndarray[any],
                    prefix_value: any,
                    padding_value: any,
                    max_sequence_length: int
                    ) -> np.ndarray[any]:
    """
    Add a prefix and pad a sequence to a given length.

    Args
        sequence: The sequence to pad.
        prefix_value: The value to prefix with.
        max_sequence_length: The length to pad to (after prefixing).
        adding_value: The value to pad with.
    Returns
        The padded sequence.
    """
    n_to_pad = max_sequence_length - len(sequence) - 1  # Subtract one for the prefix
    if n_to_pad > 0:
        sequence = np.concatenate(([prefix_value], sequence, [padding_value] * n_to_pad))
    else:
        sequence = np.concatenate(([prefix_value], sequence))
    return sequence


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

    def report_metrics(self, train: bool, objective_label: str, accelerator: Accelerator, epoch: int) -> None:
        label = "train" if train else "validation"
        label += " " + objective_label
        logging.info("Epoch %d %s mean loss: %0.2f, mean accuracy: %0.2f%%",
                     epoch,
                     label,
                     self.get_mean_loss(),
                     100 * self.get_mean_accuracy())
        performance = {f"{label} mean loss": self.get_mean_loss(),
                       f"{label} mean accuracy": self.get_mean_accuracy()}
        accelerator.log(performance, step=epoch)


def _masked_token_accuracy(token_predictions: torch.Tensor, token_ids: torch.Tensor) -> float:
    mask = (token_ids == IGNORE_INDEX)
    masked_predictions = token_predictions.argmax(2)[~mask]
    masked_labels = token_ids[~mask]
    return (masked_predictions == masked_labels).float().mean().item()


class LearningObjective(ABC):
    """
    A learning objective is a task that can be learned from the data. For example, predicting the next visit. This
    class is used to generate the data for the learning objective.
    """

    @abstractmethod
    def process_row(self, row: Dict, start_index: int, end_index: int, max_sequence_length: int) -> tuple[Dict, Dict]:
        """
        Process a row to generate input and output data for the learning objective. The start and end index indicate
        a sequence with maximum length max_sequence_length - 1 to allow prefixing with a classification token.

        Args
            row: The row to process, as generated by the CDM processing.
            start_index: Any sequence in the row should start at this index.
            end_index: Any sequence in the row should end at this index.
            max_sequence_length: The maximum length of any sequence.

        Returns
            Two dictonaries to be used by pytorch. The first is the input, the second is the output.
        """
        pass

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
    def report_performance_metrics(self, train: bool, accelerator: Accelerator, epoch: int) -> None:
        """
        Report the performance metrics.
        Args:
            train: If true, the performance metrics are for the training set. Otherwise, they are for the validation
                set.
            accelerator: The accelerator to use for logging.
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

    def process_row(self, row: Dict, start_index: int, end_index: int, max_sequence_length: int) -> tuple[Dict, Dict]:
        # Truncate the sequences:
        concept_ids = np.array(row[DataNames.CONCEPT_IDS][start_index:end_index])
        visit_segments = np.array(row[DataNames.VISIT_SEGMENTS][start_index:end_index])
        dates = np.array(row[DataNames.DATES][start_index:end_index], dtype=np.float32)
        ages = np.array(row[DataNames.AGES][start_index:end_index], dtype=np.float32)
        visit_concept_orders = np.array(row[DataNames.VISIT_CONCEPT_ORDERS][start_index:end_index])

        # Tokenize the concepts:
        token_ids = self._tokenizer.encode(concept_ids)
        # Normalize the visit_orders using the smallest visit_concept_orders. Add 1 for CLS:
        visit_concept_orders = visit_concept_orders - min(visit_concept_orders) + 1
        # Mask the tokens IDs:
        masked_token_ids, masked_token_mask = self._mask_tokens(token_ids, visit_concept_orders)

        # Prefix and pad the sequences:
        token_ids = _prefix_and_pad(sequence=token_ids,
                                    prefix_value=self._tokenizer.get_classification_token_id(),
                                    padding_value=self._tokenizer.get_padding_token_id(),
                                    max_sequence_length=max_sequence_length)
        padding_mask = _prefix_and_pad(sequence=np.zeros(shape=concept_ids.shape, dtype=bool),
                                       prefix_value=False,
                                       padding_value=True,
                                       max_sequence_length=max_sequence_length)
        masked_token_ids = _prefix_and_pad(sequence=masked_token_ids,
                                           prefix_value=self._tokenizer.get_classification_token_id(),
                                           padding_value=self._tokenizer.get_padding_token_id(),
                                           max_sequence_length=max_sequence_length)
        masked_token_mask = _prefix_and_pad(sequence=masked_token_mask,
                                            prefix_value=True,
                                            padding_value=True,
                                            max_sequence_length=max_sequence_length)
        visit_segments = _prefix_and_pad(sequence=visit_segments,
                                         prefix_value=0,
                                         padding_value=0,
                                         max_sequence_length=max_sequence_length)
        dates = _prefix_and_pad(sequence=dates,
                                prefix_value=0,
                                padding_value=0,
                                max_sequence_length=max_sequence_length)
        ages = _prefix_and_pad(sequence=ages,
                               prefix_value=0,
                               padding_value=0,
                               max_sequence_length=max_sequence_length)
        visit_concept_orders = _prefix_and_pad(sequence=visit_concept_orders,
                                               prefix_value=0,
                                               padding_value=max_sequence_length - 1,
                                               max_sequence_length=max_sequence_length)

        # Create the input and output dictionaries:
        inputs = {ModelInputNames.MASKED_TOKEN_IDS: masked_token_ids,
                  ModelInputNames.PADDING_MASK: padding_mask,
                  ModelInputNames.DATES: dates,
                  ModelInputNames.AGES: ages,
                  ModelInputNames.VISIT_SEGMENTS: visit_segments,
                  ModelInputNames.VISIT_CONCEPT_ORDERS: visit_concept_orders}
        outputs = {ModelInputNames.TOKEN_IDS: token_ids,
                   ModelInputNames.MASKED_TOKEN_MASK: masked_token_mask}
        return inputs, outputs

    def _mask_tokens(self,
                     token_ids: np.ndarray[int],
                     visit_concept_orders: np.ndarray[int]
                     ) -> tuple[np.ndarray, np.ndarray]:
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

    def report_performance_metrics(self, train: bool, accelerator: Accelerator, epoch: int) -> None:
        self._performance.report_metrics(train, "masked concept", accelerator, epoch)


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

    def process_row(self, row: Dict, start_index: int, end_index: int, max_sequence_length: int) -> tuple[Dict, Dict]:
        visit_concept_ids = np.array(row[DataNames.VISIT_CONCEPT_IDS][start_index:end_index])

        # Tokenize the visit concepts
        visit_token_ids = np.array(self._tokenizer.encode(visit_concept_ids))
        # Mask visit tokens
        masked_visit_token_ids, masked_visit_token_mask = self._mask_visit_tokens(visit_token_ids)

        # Pad the sequences
        visit_token_ids = _prefix_and_pad(sequence=visit_token_ids,
                                          prefix_value=self._tokenizer.get_classification_token_id(),
                                          padding_value=self._tokenizer.get_padding_token_id(),
                                          max_sequence_length=max_sequence_length)
        masked_visit_token_ids = _prefix_and_pad(sequence=masked_visit_token_ids,
                                                 prefix_value=self._tokenizer.get_classification_token_id(),
                                                 padding_value=self._tokenizer.get_padding_token_id(),
                                                 max_sequence_length=max_sequence_length)
        masked_visit_token_mask = _prefix_and_pad(sequence=masked_visit_token_mask,
                                                  prefix_value=True,
                                                  padding_value=True,
                                                  max_sequence_length=max_sequence_length)

        # Create the input and output dicts
        inputs = {
            ModelInputNames.MASKED_VISIT_TOKEN_IDS: masked_visit_token_ids
        }
        outputs = {
            ModelInputNames.VISIT_TOKEN_IDS: visit_token_ids,
            ModelInputNames.MASKED_VISIT_TOKEN_MASK: masked_visit_token_mask
        }
        return inputs, outputs

    def _mask_visit_tokens(self,
                           visit_token_ids: np.ndarray[int]
                           ) -> tuple[np.ndarray, np.ndarray]:
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

    def report_performance_metrics(self, train: bool, accelerator: Accelerator, epoch: int) -> None:
        self._performance.report_metrics(train, "masked visit", accelerator, epoch)


class LabelPredictionLearningObjective(LearningObjective):

    def __init__(
            self,
            concept_tokenizer: ConceptTokenizer,
            visit_tokenizer: ConceptTokenizer
    ):
        """
        Initialization
        Args:
            concept_tokenizer: The tokenizer to use to tokenize the concepts. Should already be trained.
               visit_tokenizer: The tokenizer to use to tokenize the visits. Should already be trained.
        """
        self._concept_tokenizer = concept_tokenizer
        self._visit_tokenizer = visit_tokenizer
        self._criterion = torch.nn.CrossEntropyLoss()
        self._performance = TokenPredictionPerformance()

    def process_row(self, row: Dict, start_index: int, end_index: int, max_sequence_length: int) -> tuple[Dict, Dict]:
        # Truncate the sequences:
        concept_ids = np.array(row[DataNames.CONCEPT_IDS][start_index:end_index])
        visit_segments = np.array(row[DataNames.VISIT_SEGMENTS][start_index:end_index])
        dates = np.array(row[DataNames.DATES][start_index:end_index], dtype=np.float32)
        ages = np.array(row[DataNames.AGES][start_index:end_index], dtype=np.float32)
        visit_concept_orders = np.array(row[DataNames.VISIT_CONCEPT_ORDERS][start_index:end_index])
        visit_concept_ids = np.array(row[DataNames.VISIT_CONCEPT_IDS][start_index:end_index])
        label = row[DataNames.LABEL]

        # Tokenize:
        token_ids = self._concept_tokenizer.encode(concept_ids)
        visit_token_ids = np.array(self._visit_tokenizer.encode(visit_concept_ids))

        # Normalize the visit_orders using the smallest visit_concept_orders. Add 1 for CLS:
        visit_concept_orders = visit_concept_orders - min(visit_concept_orders) + 1

        # Prefix and pad the sequences:
        token_ids = _prefix_and_pad(sequence=token_ids,
                                    prefix_value=self._concept_tokenizer.get_classification_token_id(),
                                    padding_value=self._concept_tokenizer.get_padding_token_id(),
                                    max_sequence_length=max_sequence_length)
        padding_mask = _prefix_and_pad(sequence=np.zeros(shape=concept_ids.shape, dtype=bool),
                                       prefix_value=False,
                                       padding_value=True,
                                       max_sequence_length=max_sequence_length)
        visit_segments = _prefix_and_pad(sequence=visit_segments,
                                         prefix_value=0,
                                         padding_value=0,
                                         max_sequence_length=max_sequence_length)
        dates = _prefix_and_pad(sequence=dates,
                                prefix_value=0,
                                padding_value=0,
                                max_sequence_length=max_sequence_length)
        ages = _prefix_and_pad(sequence=ages,
                               prefix_value=0,
                               padding_value=0,
                               max_sequence_length=max_sequence_length)
        visit_concept_orders = _prefix_and_pad(sequence=visit_concept_orders,
                                               prefix_value=0,
                                               padding_value=max_sequence_length - 1,
                                               max_sequence_length=max_sequence_length)
        visit_token_ids = _prefix_and_pad(sequence=visit_token_ids,
                                          prefix_value=self._visit_tokenizer.get_classification_token_id(),
                                          padding_value=self._visit_tokenizer.get_padding_token_id(),
                                          max_sequence_length=max_sequence_length)

        # Create the input and output dictionaries:
        inputs = {ModelInputNames.TOKEN_IDS: token_ids,
                  ModelInputNames.PADDING_MASK: padding_mask,
                  ModelInputNames.DATES: dates,
                  ModelInputNames.AGES: ages,
                  ModelInputNames.VISIT_SEGMENTS: visit_segments,
                  ModelInputNames.VISIT_CONCEPT_ORDERS: visit_concept_orders,
                  ModelInputNames.VISIT_TOKEN_IDS: visit_token_ids}
        outputs = {ModelInputNames.FINETUNE_LABEL: label}
        return inputs, outputs

    def compute_loss(self, outputs: Dict[str, torch.Tensor], predictions: Dict[str, torch.Tensor]) -> torch.Tensor:
        label_predictions = predictions[ModelOutputNames.LABEL_PREDICTIONS]
        labels = outputs[ModelInputNames.FINETUNE_LABEL].long()

        loss = self._criterion(label_predictions, labels)
        visit_token_accuracy = _masked_token_accuracy(label_predictions.unsqueeze(1), labels.unsqueeze(1))
        self._performance.add(loss=loss.float().mean().item(),
                              accuracy=visit_token_accuracy)
        return loss

    def reset_performance_metrics(self) -> None:
        self._performance.reset()

    def report_performance_metrics(self, train: bool, accelerator: Accelerator, epoch: int) -> None:
        self._performance.report_metrics(train, "label prediction", accelerator, epoch)
