"""
Simulate CDM data for testing purposes. Persons are simulated to have hidden disease states. Some states are fixes at
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

import numpy as np
import tqdm as tqdm

import cdm_data
from cdm_data import CdmData
import utils.logger as logger

LOGGER_FILE_NAME = "_simulation_log.txt"
QUIET = 0
CHATTY = 1
OBSESSIVE = 2
JSON_FILE_NAME = "_simulation.json"


def logistic(x):
    return 1 / (1 + np.exp(-x))


class SimulationSettings:
    """
    Settings for the simulation.
    """

    def __init__(self,
                 dynamic_state_count: int = 10,
                 fixed_state_count: int = 5,
                 concept_count: int = 100,
                 serious_concept_count: int = 10,
                 visit_multiplier: int = 2,
                 days_to_simulate: int = 365 * 2):
        self.dynamic_state_count = dynamic_state_count
        self.fixed_state_count = fixed_state_count
        self.concept_count = concept_count
        self.serious_concept_count = serious_concept_count
        self.visit_multiplier = visit_multiplier
        self.days_to_simulate = days_to_simulate


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


class Simulation:
    # Note: Tried using scipy.sparse.csr_matrix, but it was 100 times slower than using a dense matrix.

    def __init__(self,
                 settings: SimulationSettings,
                 output_path: str,
                 max_cores: int = 1,
                 log_verbosity: int = QUIET,
                 partition_count: int = 50,
                 person_count: int = 1000):
        self._settings = settings
        self._state_count = settings.dynamic_state_count + settings.fixed_state_count + 2  # + 2 for age and sex
        self._concept_ids = np.arange(settings.concept_count) + 1000000
        self._cdm_data = CdmData()
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        self._output_path = output_path
        self._max_cores = max_cores
        self._profile = False
        self._log_verbosity = log_verbosity
        self._configure_logger()
        self._partition_count = partition_count
        self._person_count = person_count

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
        self._serious_concepts = np.random.choice(self._concept_ids, size=settings.serious_concept_count)
        self.save_to_json(os.path.join(self._output_path, JSON_FILE_NAME))

    def set_profile(self, profile: bool):
        self._profile = profile

    def get_profile(self):
        return self._profile

    def _configure_logger(self):
        logger.create_logger(os.path.join(self._output_path, LOGGER_FILE_NAME))

    def _simulate_person(self, person_id: int):
        start_date = np.datetime64("2010-01-01") + np.random.randint(0, 365 * 10)
        state = np.random.binomial(n=1, p=self._initial_state_probabilities)
        self._cdm_data.add_observation_period(person_id=person_id,
                                              observation_period_id=person_id,
                                              observation_period_start_date=start_date,
                                              observation_period_end_date=start_date + self._settings.days_to_simulate)
        year_of_birth, month_of_birth, day_of_birth = _simulate_date_of_birth(start_date, state[-2])
        gender_concept_id = 8507 if state[-1] == 0 else 8532
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
            state_flip_probabilities = np.zeros(shape=self._settings.dynamic_state_count)
            # Role dice to enter a dynamic state:
            for i in range(self._settings.dynamic_state_count):
                if state[i] == 1:
                    continue
                value = -6 + (state_interaction_matrix * self._dynamic_state_entry_coefs[i]).sum()
                state_flip_probabilities[i] = logistic(value)
            # Role dice to exit a dynamic state:
            for i in range(self._settings.dynamic_state_count):
                if state[i] == 0:
                    continue
                state_flip_probabilities[i] = self._dynamic_state_exit_probabilities[i]
            state[:self._settings.dynamic_state_count] = np.logical_xor(state[:self._settings.dynamic_state_count],
                                                                        np.random.binomial(n=1,
                                                                                           p=state_flip_probabilities))
            # Role dice to observe a concept:
            concept_probabilities = np.zeros(shape=self._settings.concept_count)
            for i in range(self._settings.concept_count):
                value = -8 + (state_interaction_matrix * self._concept_emmision_coefs[i]).sum()
                concept_probabilities[i] = logistic(value)
            observed_concepts = np.random.binomial(n=1, p=concept_probabilities)
            visit = sum(observed_concepts) > 0
            if visit:
                # If admission concept is serious, make the visit an emergency room visit, otherwise make it an
                # outpatient visit:
                if np.in1d(self._concept_ids[observed_concepts != 0], self._serious_concepts).any():
                    visit_concept_id = 9203  # Emergency room visit
                else:
                    visit_concept_id = 9201  # Outpatient visit
                # Role dice again to observe a concept (to simulate the fact that some concepts are more likely to be
                # observed during a visit):
                observed_concepts = observed_concepts | np.random.binomial(n=self._settings.visit_multiplier,
                                                                           p=concept_probabilities)
                concept_ids = self._concept_ids[observed_concepts != 0]
                for concept_id in concept_ids:
                    self._cdm_data.add_condition_occurrence(person_id=person_id,
                                                            condition_start_date=start_date + t,
                                                            condition_concept_id=concept_id,
                                                            visit_occurrence_id=visit_occurrence_id)
                self._cdm_data.add_visit_occurrence(person_id=person_id,
                                                    visit_start_date=start_date + t,
                                                    visit_end_date=start_date + t,
                                                    visit_concept_id=visit_concept_id,
                                                    visit_occurrence_id=visit_occurrence_id)
                visit_occurrence_id += 1
                if self._log_verbosity == OBSESSIVE:
                    logging.debug("Person %s visit on day %s with concept IDs: %s", person_id, t, concept_ids)

    def simulate(self):
        """
        Process the CDM data in the provided cdm_data_path.
        """
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
            for partition_i in range(self._partition_count):
                self._simulate_partition(partition_i)
        else:
            pool = multiprocessing.get_context("spawn").Pool(processes=self._max_cores)
            tasks = range(self._partition_count)
            work = self._simulate_partition
            for _ in tqdm.tqdm(pool.imap_unordered(work, tasks), total=len(tasks)):
                pass
            pool.close()
        logging.info("Finished simulation")

    def _simulate_partition(self, partition_i: int):
        # This function is executed within a thread
        # Need to re-configure logger because we're in a thread:
        self._configure_logger()
        logging.debug("Starting partition %s of %s", partition_i, self._partition_count)
        persons_per_partition = self._person_count // self._partition_count
        for i in range(persons_per_partition * partition_i, persons_per_partition * (partition_i + 1)):
            self._simulate_person(person_id=i)
        self._cdm_data.log_statistics(partition_i=partition_i)
        self._cdm_data.write_to_parquet(self._output_path, partition_i)
        self._cdm_data = cdm_data.CdmData()
        logging.debug("Finished partition %s of %s", partition_i, self._partition_count)

    def save_to_json(self, file_name: str):
        with open(file_name, "w") as f:
            to_save = {
                "simulation_settings": self._settings.__dict__,
                "state_count": self._state_count,
                "concept_ids": self._concept_ids.tolist(),
                "serious_concepts": self._serious_concepts.tolist(),
                "initial_state_probabilities": self._initial_state_probabilities.tolist(),
                "dynamic_state_exit_probabilities": self._dynamic_state_exit_probabilities.tolist(),
                "dynamic_state_entry_coefs": [matrix.tolist() for matrix in self._dynamic_state_entry_coefs],
                "concept_emmision_coefs": [matrix.tolist() for matrix in self._concept_emmision_coefs]
            }
            json.dump(to_save, f)


def create_from_json(file_name: str,
                     output_path: str,
                     max_cores: int = 1,
                     log_verbosity: int = QUIET,
                     partition_count: int = 1,
                     person_count: int = 1000):
    with open(file_name, "r") as f:
        loaded = json.load(f)
    simulation_settings = SimulationSettings(**loaded["simulation_settings"])
    simulation = Simulation(settings=simulation_settings,
                            output_path=output_path,
                            max_cores=max_cores,
                            log_verbosity=log_verbosity,
                            partition_count=partition_count,
                            person_count=person_count)
    simulation._state_count = loaded["state_count"]
    simulation._concept_ids = np.array(loaded["concept_ids"])
    simulation._serious_concepts = np.array(loaded["serious_concepts"])
    simulation._initial_state_probabilities = np.array(loaded["initial_state_probabilities"])
    simulation._dynamic_state_entry_coefs = [np.array(matrix) for matrix in loaded["dynamic_state_entry_coefs"]]
    simulation._dynamic_state_exit_probabilities = np.array(loaded["dynamic_state_exit_probabilities"])
    simulation._concept_emmision_coefs = [np.array(matrix) for matrix in loaded["concept_emmision_coefs"]]
    return simulation


def main(args: List[str]):
    config = configparser.ConfigParser()
    config.read(args[0])
    if config["simulation"].get("json_file_name") == "":
        simulation_settings = SimulationSettings(
            dynamic_state_count=config["simulation"].getint("dynamic_state_count"),
            fixed_state_count=config["simulation"].getint("fixed_state_count"),
            concept_count=config["simulation"].getint("concept_count"),
            serious_concept_count=config["simulation"].getint("serious_concept_count"),
            visit_multiplier=config["simulation"].getint("visit_multiplier"),
            days_to_simulate=config["simulation"].getint("days_to_simulate"))
        simulation = Simulation(simulation_settings,
                                output_path=config["system"].get("output_path"),
                                max_cores=config["system"].getint("max_cores"),
                                log_verbosity=config["debug"].getint("log_verbosity"),
                                partition_count=config["execution"].getint("partition_count"),
                                person_count=config["execution"].getint("person_count"))
    else:
        simulation = create_from_json(file_name=config["simulation"].get("json_file_name"),
                                      output_path=config["system"].get("output_path"),
                                      max_cores=config["system"].getint("max_cores"),
                                      log_verbosity=config["debug"].getint("log_verbosity"),
                                      partition_count=config["execution"].getint("partition_count"),
                                      person_count=config["execution"].getint("person_count"))
        # Log config after initializing cdm_data_processor so logger is initialized:
        logging.info("Loaded simulation configuration from %s", config["simulation"].get("json_file_name"))
        config.remove_section("simulation")
    logger.log_config(config)
    if config["debug"].getboolean("profile"):
        simulation.set_profile(True)
    simulation.simulate()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise Exception("Must provide path to ini file as argument")
    else:
        main(sys.argv[1:])
