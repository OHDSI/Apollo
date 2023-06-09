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
import logging
import multiprocessing
import os

import numpy as np
import tqdm as tqdm

import cdm_data
from cdm_data import CdmData
import utils.logger as logger

LOGGER_FILE_NAME = "_simulation_log.txt"
QUIET = 0
CHATTY = 1
OBSESSIVE = 2


def logistic(x):
    return 1 / (1 + np.exp(-x))


class SimulationSettings:
    """
    Settings for the simulation.
    """

    def __init__(self):
        self.partition_count = 50
        self.person_count = 10000
        self.dynamic_state_count = 10
        self.fixed_state_count = 5
        self.concept_count = 100
        self.visit_multiplier = 2
        self.days_to_simulate = 365 * 2


class Simulation:
    # Note: Tried using scipy.sparse.csr_matrix, but it was 100 times slower than using a dense matrix.

    def __init__(self, settings: SimulationSettings, output_path: str, max_cores: int = 1, log_verbosity: int = QUIET):
        self._settings = settings
        self._state_count = settings.dynamic_state_count + settings.fixed_state_count + 2  # + 2 for age and sex
        self._concept_ids = np.arange(settings.concept_count) + 1000000
        self._cdm_data = CdmData()
        self._output_path = output_path
        self._max_cores = max_cores
        self._profile = False
        self._log_verbosity = log_verbosity
        self._configure_logger()

        # Initialize probabilities and coefficients:
        self._initial_state_probabilities = np.concatenate((
            np.random.uniform(size=settings.dynamic_state_count, high=0.01),
            np.random.uniform(size=settings.fixed_state_count + 2)
        ))
        # Note: really only need half of these matrices to be filled, because interaction matrix will be symmetrical 
        # over diagonal, but it's easier to just fill them completely:
        self._dynamic_state_entry_coefs = []
        for i in range(settings.dynamic_state_count):
            matrix = np.zeros(self._state_count * self._state_count)
            non_zero_count = round(len(matrix) * 0.01)
            matrix[np.random.choice(len(matrix), size=non_zero_count, replace=False)] = \
                np.random.laplace(loc=0, scale=0.1, size=non_zero_count)
            self._dynamic_state_entry_coefs.append(matrix.reshape(self._state_count, self._state_count))
        self._dynamic_state_exit_probabilities = np.random.uniform(size=settings.dynamic_state_count, high=0.1)
        self._concept_emmision_coefs = []
        for i in range(settings.concept_count):
            matrix = np.zeros(self._state_count * self._state_count)
            non_zero_count = round(len(matrix) * 0.01)
            matrix[np.random.choice(len(matrix), size=non_zero_count, replace=False)] = \
                np.random.laplace(loc=0, scale=0.1, size=non_zero_count)
            self._concept_emmision_coefs.append(matrix.reshape(self._state_count, self._state_count))

    def set_profile(self, profile: bool):
        self._profile = profile

    def get_profile(self):
        return self._profile

    def _configure_logger(self):
        logger.create_logger(os.path.join(self._output_path, LOGGER_FILE_NAME))

    def _simulate_date_of_birth(self, current_date: np.datetime64, age_indicator: int) -> tuple[int, int, int]:
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

    def _simulate_person(self, person_id: int):
        start_date = np.datetime64("2010-01-01") + np.random.randint(0, 365 * 10)
        self._cdm_data.add_observation_period(person_id=person_id,
                                              observation_period_id=person_id,
                                              observation_period_start_date=start_date,
                                              observation_period_end_date=start_date + self._settings.days_to_simulate)
        state = np.random.binomial(n=1, p=self._initial_state_probabilities)

        year_of_birth, month_of_birth, day_of_birth = self._simulate_date_of_birth(start_date, state[-2])
        gender_concept_id = 8507 if state[-1] == 0 else 8532
        self._cdm_data.add_person(person_id=person_id,
                                  year_of_birth=year_of_birth,
                                  month_of_birth=month_of_birth,
                                  day_of_birth=day_of_birth,
                                  gender_concept_id=gender_concept_id)

        visit_occurrence_id = person_id * 100000

        # state = sparse.csr_matrix(state)
        for t in range(self._settings.days_to_simulate):
            if self._log_verbosity == OBSESSIVE:
                logging.debug(f"Person {person_id}, Day {t}, state: {state}")
            # Role dice to enter a dynamic state:
            state_interaction_matrix = state.transpose() * state
            for i in range(self._settings.dynamic_state_count):
                if state[i] == 1:
                    continue
                value = -3 + (state_interaction_matrix * self._dynamic_state_entry_coefs[i]).sum()
                value = logistic(value)
                if np.random.binomial(n=1, p=value):
                    state[i] = 1
            # Role dice to exit a dynamic state:
            for i in range(self._settings.dynamic_state_count):
                if state[i] == 0:
                    continue
                value = self._dynamic_state_exit_probabilities[i]
                if np.random.binomial(n=1, p=value):
                    state[i] = 0
            # Role dice to observe a concept:
            concept_probabilities = np.zeros(shape=self._settings.concept_count)
            for i in range(self._settings.concept_count):
                value = - 6 + (state_interaction_matrix * self._concept_emmision_coefs[i]).sum()
                concept_probabilities[i] = logistic(value)
            observed_concepts = np.random.binomial(n=1, p=concept_probabilities)
            visit = sum(observed_concepts) > 0
            if visit:
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
                                                    visit_concept_id=9201,
                                                    visit_occurrence_id=visit_occurrence_id)
                visit_occurrence_id += 1
                if self._log_verbosity == OBSESSIVE:
                    logging.debug(f"Person {person_id} visit on day {t} with concept IDs: {concept_ids}")

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
            for partition_i in range(self._settings.partition_count):
                self._simulate_partition(partition_i)
        else:
            pool = multiprocessing.get_context("spawn").Pool(processes=self._max_cores)
            tasks = range(self._settings.partition_count)
            work = self._simulate_partition
            for _ in tqdm.tqdm(pool.imap_unordered(work, tasks), total=len(tasks)):
                pass
            pool.close()
        logging.info("Finished simulation")

    def _simulate_partition(self, partition_i: int):
        # This function is executed within a thread
        # Need to re-configure logger because we're in a thread:
        self._configure_logger()
        logging.debug("Starting partition %s of %s", partition_i, self._settings.partition_count)
        persons_per_partition = self._settings.person_count // self._settings.partition_count
        for i in range(persons_per_partition * partition_i, persons_per_partition * (partition_i + 1)):
            self._simulate_person(person_id=i)
        self._cdm_data.log_statistics(partition_i=partition_i)
        self._cdm_data.write_to_parquet(self._output_path, partition_i)
        self._cdm_data = cdm_data.CdmData()
        logging.debug("Finished partition %s of %s", partition_i, self._settings.partition_count)


if __name__ == "__main__":
    sim_settings = SimulationSettings()
    simulation = Simulation(sim_settings, output_path="d:/GPM_Sim", max_cores=25, log_verbosity=QUIET)
    simulation.simulate()
