"""
Simulate CDM data for testing purposes. Persons are simulated to have hidden disease states. Some states are fixed at
the start, to simulate fixed traits such as genetics, while the remaining states are dynamic. The probability to enter
a dynamic disease state depending on the current disease states in a non-linear way (i.e. using interactions between
states). Concepts are simulated to be observed with probabilities depending on the current disease states, again in a
non-linear way. Concepts imply visits, and visits in return increase the probability of observing concepts, thus causing
concepts to cluster in time. The simulated data is saved in the CDM format.

To generate patient-level predicition problems with a gold standard, the model is executed. For each person, an index
date is randomly selected, and Monte-Carlo simulations are performed to simulate the future.
"""
import cProfile
import configparser
import json
import logging
import multiprocessing
import os
import sys
from typing import List
from dataclasses import dataclass

import numpy as np
import tqdm as tqdm

import cdm_data
import utils.logger as logger

LOGGER_FILE_NAME = "_simulation_log.txt"
QUIET = 0
CHATTY = 1
OBSESSIVE = 2
JSON_FILE_NAME = "_simulation.json"
PRETRAINING_FOLDER = "pretraining"
TRAIN_FOLDER = "train"
TEST_FOLDER = "test"
STATE_TRANSITION_INTERCEPT = -6  # Log probability intercept of transitioning to a new state.
CONCEPT_OBSERVATION_INTERCEPT = -8  # Log probability intercept of observing a concept.
AGE_INDEX = -2
GENDER_INDEX = -1


def logistic(x):
    return 1 / (1 + np.exp(-x))


@dataclass
class SimulationSettings:
    dynamic_state_count: int = 10
    fixed_state_count: int = 5
    concept_count: int = 100
    serious_concept_count: int = 10
    visit_multiplier: int = 2
    days_to_simulate: int = 365 * 2


@dataclass
class SimulationTask:
    partition_count: int


@dataclass
class PreTrainingTask(SimulationTask):
    person_count: int


@dataclass
class PredictionTask(SimulationTask):
    train_person_count: int
    test_person_count: int
    prediction_window: int


def _simulate_date_of_birth(current_date: np.datetime64, age_indicator: int) -> tuple[int, int, int]:
    """
    Simulate a date of birth for a person.

    Args:
        current_date: The current date.
        age_indicator: An indicator of the age of the person.
    Returns:
        A tuple of the year, month and day of birth.
    """
    if age_indicator == 0:
        # Simulate a date of birth for a child:
        birth_date = current_date - np.random.randint(0, 365 * 18)
    else:
        # Simulate a date of birth for an adult:
        birth_date = current_date - np.random.randint(365 * 18, 365 * 100)
    birth_date = np.datetime64(birth_date, 'D')
    return birth_date.astype('datetime64[Y]').astype(int) + 1970, \
           birth_date.astype('datetime64[M]').astype(int) % 12 + 1, \
           birth_date.astype('datetime64[D]').astype(int) % 31 + 1


class Simulator:
    # Note: Tried using scipy.sparse.csr_matrix, but it was 100 times slower than using a dense matrix.

    def __init__(self,
                 root_folder: str,
                 settings: SimulationSettings = None,
                 json_file_name: str = None,
                 log_verbosity: int = QUIET,
                 max_cores: int = 1):
        self._root_folder = root_folder
        if not os.path.exists(root_folder):
            os.makedirs(root_folder)
        self._profile = False
        self._log_verbosity = log_verbosity
        self._max_cores = max_cores
        self._task = SimulationTask(0)
        self._configure_logger()
        if settings is None:
            self._init_from_json(json_file_name)
        else:
            self._init_from_settings(settings)
            self.save_to_json(os.path.join(self._root_folder, JSON_FILE_NAME))

    def _init_from_json(self, json_file_name: str):
        logging.info("Loading simulation configuration from %s", json_file_name)
        with open(json_file_name, "r") as f:
            loaded = json.load(f)
        self._settings = SimulationSettings(**loaded["simulation_settings"])
        self._state_count = loaded["state_count"]
        self._concept_ids = np.array(loaded["concept_ids"])
        self._serious_concept_idx = np.array(loaded["serious_concepts"])
        self._initial_state_probabilities = np.array(loaded["initial_state_probabilities"])
        self._dynamic_state_entry_coefs = [np.array(matrix) for matrix in loaded["dynamic_state_entry_coefs"]]
        self._dynamic_state_exit_probabilities = np.array(loaded["dynamic_state_exit_probabilities"])
        self._concept_emmision_coefs = [np.array(matrix) for matrix in loaded["concept_emmision_coefs"]]

    def _init_from_settings(self, settings: SimulationSettings):
        self._settings = settings
        self._state_count = settings.dynamic_state_count + settings.fixed_state_count + 2  # + 2 for age and sex
        self._concept_ids = np.arange(settings.concept_count) + 1000000

        # Initialize probabilities and coefficients:
        self._initial_state_probabilities = np.concatenate((
            np.random.beta(size=settings.dynamic_state_count, a=1, b=4) / settings.dynamic_state_count,
            np.random.uniform(size=settings.fixed_state_count + 2)
        ))
        # Note: really only need half of these matrices to be filled, because interaction matrix will be symmetrical
        # over diagonal, but it's easier to just fill them completely:
        self._dynamic_state_entry_coefs = []
        for i in range(settings.dynamic_state_count):
            matrix = np.zeros(self._state_count * self._state_count)
            non_zero_count = round(len(matrix) / self._state_count)
            matrix[np.random.choice(len(matrix), size=non_zero_count, replace=False)] = \
                np.random.laplace(loc=0, scale=0.1, size=non_zero_count)
            self._dynamic_state_entry_coefs.append(matrix.reshape(self._state_count, self._state_count))
        self._dynamic_state_exit_probabilities = \
            np.random.beta(size=settings.dynamic_state_count, a=0.2, b=3) / settings.dynamic_state_count
        self._concept_emmision_coefs = []
        for i in range(settings.concept_count):
            matrix = np.zeros(self._state_count * self._state_count)
            non_zero_count = round(len(matrix) / np.sqrt(self._settings.concept_count))
            matrix[np.random.choice(len(matrix), size=non_zero_count, replace=False)] = \
                np.random.laplace(loc=0, scale=0.5, size=non_zero_count)
            self._concept_emmision_coefs.append(matrix.reshape(self._state_count, self._state_count))
        self._serious_concept_idx = np.zeros(settings.concept_count, dtype=bool)
        self._serious_concept_idx[np.random.choice(settings.concept_count,
                                                   size=settings.serious_concept_count,
                                                   replace=False)] = True

    def set_profile(self, profile: bool):
        self._profile = profile

    def get_profile(self):
        return self._profile

    def _configure_logger(self):
        logger.create_logger(os.path.join(self._root_folder, LOGGER_FILE_NAME))

    def _simulate_person(self, person_id: int):
        if isinstance(self._task, PredictionTask):
            prediction_labels = np.zeros(self._settings.concept_count, dtype=bool)
            # Currently just using full prediction window, but could change to make index day random:
            index_day = self._settings.days_to_simulate - self._task.prediction_window
            is_prediction = True
        else:
            prediction_labels = None
            index_day = 0
            is_prediction = False
        start_date = np.datetime64("2010-01-01") + np.random.randint(0, 365 * 10)
        state = np.random.binomial(n=1, p=self._initial_state_probabilities)
        self._cdm_data.add_observation_period(person_id=person_id,
                                              observation_period_id=person_id,
                                              observation_period_start_date=start_date,
                                              observation_period_end_date=start_date + self._settings.days_to_simulate)
        year_of_birth, month_of_birth, day_of_birth = _simulate_date_of_birth(start_date, state[AGE_INDEX])
        gender_concept_id = 8507 if state[GENDER_INDEX] == 0 else 8532
        self._cdm_data.add_person(person_id=person_id,
                                  year_of_birth=year_of_birth,
                                  month_of_birth=month_of_birth,
                                  day_of_birth=day_of_birth,
                                  gender_concept_id=gender_concept_id)
        visit_occurrence_id = person_id * 100000
        for t in range(self._settings.days_to_simulate):
            if self._log_verbosity == OBSESSIVE:
                logging.debug("Person: %s, Day: %s, State: %s", person_id, t, state)
            state_interaction_matrix = np.outer(state, state)
            # Roll dice to change state:
            flip_to_one = logistic(
                np.sum(np.asarray(self._dynamic_state_entry_coefs) * state_interaction_matrix[np.newaxis, :, :],
                       axis=(1, 2)) - 6) * (state[:self._settings.dynamic_state_count] == 0)
            flip_to_zero = (self._dynamic_state_exit_probabilities * (state[:self._settings.dynamic_state_count] == 1))
            state_flip_probabilities = flip_to_one + flip_to_zero
            state[:self._settings.dynamic_state_count] = np.logical_xor(state[:self._settings.dynamic_state_count],
                                                                        np.random.binomial(n=1,
                                                                                           p=state_flip_probabilities))
            # Roll dice to observe a concept:
            concept_probabilities = logistic(CONCEPT_OBSERVATION_INTERCEPT + np.sum(
                np.asarray(self._concept_emmision_coefs) * state_interaction_matrix[np.newaxis, :, :], axis=(1, 2)))
            admission_concept_idx = (np.random.binomial(n=1, p=concept_probabilities) != 0)
            visit = admission_concept_idx.any()
            if visit:
                # Roll dice again to observe a concept (to simulate the fact that some concepts are more likely to be
                # observed during a visit):
                observed_concept_idx = admission_concept_idx | (np.random.binomial(n=self._settings.visit_multiplier,
                                                                                   p=concept_probabilities) != 0)
                if is_prediction and t > index_day:
                    prediction_labels = prediction_labels | observed_concept_idx
                else:
                    concept_ids = self._concept_ids[observed_concept_idx]
                    for concept_id in concept_ids:
                        self._cdm_data.add_condition_occurrence(person_id=person_id,
                                                                condition_start_date=start_date + t,
                                                                condition_concept_id=concept_id,
                                                                visit_occurrence_id=visit_occurrence_id)
                    # If admission concept is serious, make the visit an emergency room visit, otherwise make it an
                    # outpatient visit:
                    if (admission_concept_idx & self._serious_concept_idx).any():
                        visit_concept_id = 9203  # Emergency room visit
                    else:
                        visit_concept_id = 9201  # Outpatient visit
                    self._cdm_data.add_visit_occurrence(person_id=person_id,
                                                        visit_start_date=start_date + t,
                                                        visit_end_date=start_date + t,
                                                        visit_concept_id=visit_concept_id,
                                                        visit_occurrence_id=visit_occurrence_id)
                    visit_occurrence_id += 1
                    if self._log_verbosity == OBSESSIVE:
                        logging.debug("Person %s visit on day %s with concept IDs: %s", person_id, t, concept_ids)
        if isinstance(self._cdm_data, cdm_data.CdmDataWithLabels):
            for i in range(self._settings.concept_count):
                self._cdm_data.add_label(person_id=person_id,
                                         concept_id=self._concept_ids[i],
                                         label=prediction_labels[i])

    def simulate(self, task: SimulationTask):
        """
        Process the CDM data in the provided cdm_data_path.
        """
        self._task = task
        if isinstance(task, PreTrainingTask):
            logging.info("Simulating data for pre-training task")
            output_folder = os.path.join(self._root_folder, PRETRAINING_FOLDER)
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
        elif isinstance(task, PredictionTask):
            logging.info("Simulating data for prediction task")
            train_folder = os.path.join(self._root_folder, TRAIN_FOLDER)
            if not os.path.exists(train_folder):
                os.makedirs(train_folder)
            test_folder = os.path.join(self._root_folder, TEST_FOLDER)
            if not os.path.exists(test_folder):
                os.makedirs(test_folder)
        else:
            raise ValueError("Unknown task type: %s" % type(task))
        if self._profile:
            cProfile.runctx(statement="self._simulate()",
                            locals={"self": self},
                            globals={},
                            filename="../stats")
        else:
            self._simulate()

    def _simulate(self):
        if self._profile:
            logging.info("Profiling mode enabled, running first partition in single thread")
            self._simulate_partition(0)
        elif self._max_cores == 1:
            # Run single thread in main thread for easier debugging:
            for partition_i in range(self._task.partition_count):
                self._simulate_partition(partition_i)
        else:
            pool = multiprocessing.get_context("spawn").Pool(processes=self._max_cores)
            tasks = range(self._task.partition_count)
            work = self._simulate_partition
            for _ in tqdm.tqdm(pool.imap_unordered(work, tasks), total=len(tasks)):
                pass
            pool.close()
        logging.info("Finished simulation")

    def _simulate_partition(self, partition_i: int):
        # This function is executed within a thread
        # Need to re-configure logger because we're in a thread:
        self._configure_logger()
        logging.debug("Starting partition %s of %s", partition_i, self._task.partition_count)
        if isinstance(self._task, PreTrainingTask):
            self._simulate_person_set(partition_i=partition_i,
                                      persons_per_partition=self._task.person_count // self._task.partition_count,
                                      output_folder=os.path.join(self._root_folder, PRETRAINING_FOLDER))
        elif isinstance(self._task, PredictionTask):
            self._simulate_person_set(partition_i=partition_i,
                                      persons_per_partition=self._task.train_person_count // self._task.partition_count,
                                      output_folder=os.path.join(self._root_folder, TRAIN_FOLDER))
            self._simulate_person_set(partition_i=partition_i,
                                      persons_per_partition=self._task.test_person_count // self._task.partition_count,
                                      output_folder=os.path.join(self._root_folder, TEST_FOLDER))
        else:
            raise ValueError("Unknown task type: %s" % type(self._task))
        logging.debug("Finished partition %s of %s", partition_i, self._task.partition_count)

    def _simulate_person_set(self,
                             partition_i: int,
                             persons_per_partition: int,
                             output_folder: str):
        if isinstance(self._task, PredictionTask):
            self._cdm_data = cdm_data.CdmDataWithLabels()
        else:
            self._cdm_data = cdm_data.CdmData()
        for i in range(persons_per_partition * partition_i, persons_per_partition * (partition_i + 1)):
            self._simulate_person(person_id=i)
        self._cdm_data.log_statistics(partition_i=partition_i)
        self._cdm_data.write_to_parquet(output_folder, partition_i)

    def save_to_json(self, file_name: str):
        with open(file_name, "w") as f:
            to_save = {
                "simulation_settings": self._settings.__dict__,
                "state_count": self._state_count,
                "concept_ids": self._concept_ids.tolist(),
                "serious_concept_idx": self._serious_concept_idx.tolist(),
                "initial_state_probabilities": self._initial_state_probabilities.tolist(),
                "dynamic_state_exit_probabilities": self._dynamic_state_exit_probabilities.tolist(),
                "dynamic_state_entry_coefs": [matrix.tolist() for matrix in self._dynamic_state_entry_coefs],
                "concept_emmision_coefs": [matrix.tolist() for matrix in self._concept_emmision_coefs]
            }
            json.dump(to_save, f)


def main(args: List[str]):
    config = configparser.ConfigParser()
    with open(args[0]) as file:  # Explicitly opening file so error is thrown when not found
        config.read_file(file)
    if config["simulation"].get("json_file_name") == "":
        settings = SimulationSettings(
            dynamic_state_count=config["simulation"].getint("dynamic_state_count"),
            fixed_state_count=config["simulation"].getint("fixed_state_count"),
            concept_count=config["simulation"].getint("concept_count"),
            serious_concept_count=config["simulation"].getint("serious_concept_count"),
            visit_multiplier=config["simulation"].getint("visit_multiplier"),
            days_to_simulate=config["simulation"].getint("days_to_simulate"))
        json_file_name = None
    else:
        settings = None
        json_file_name = config["simulation"].get("json_file_name")
        config.remove_section("simulation")
    simulator = Simulator(root_folder=config["system"].get("root_folder"),
                          settings=settings,
                          json_file_name=json_file_name,
                          log_verbosity=config["debug"].getint("log_verbosity"),
                          max_cores=config["system"].getint("max_cores"))
    # Log config after initializing cdm_data_processor so logger is initialized:
    logger.log_config(config)
    if config["debug"].getboolean("profile"):
        simulator.set_profile(True)
    if config["pre-training data generation"].getboolean("generate_pre_training_data"):
        task = PreTrainingTask(partition_count=config["pre-training data generation"].getint("partition_count"),
                               person_count=config["pre-training data generation"].getint("person_count"))
        simulator.simulate(task)
    if config["prediction data generation"].getboolean("generate_prediction_data"):
        task = PredictionTask(partition_count=config["prediction data generation"].getint("partition_count"),
                              train_person_count=config["prediction data generation"].getint("train_person_count"),
                              test_person_count=config["prediction data generation"].getint("test_person_count"),
                              prediction_window=config["prediction data generation"].getint("prediction_window"))
        simulator.simulate(task)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise Exception("Must provide path to ini file as argument")
    else:
        main(sys.argv[1:])
