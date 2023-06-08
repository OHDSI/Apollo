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
from typing import Type

import numpy
import numpy as np


def logistic(x):
    return 1 / (1 + np.exp(-x))


class DynamicArray:
    """
    A dynamic array that can be appended to one at a time. The array is split into blocks of a fixed size, and the
    blocks are concatenated when the array is collected.
    """

    _sub_array_size = 1000

    def __init__(self, dtype: Type = np.float64):
        self._dtype = dtype
        self._blocks = []
        self._current_block = self._create_block()
        self._cursor = -1

    def _create_block(self):
        if self._dtype == np.datetime64:
            return np.full(shape=self._sub_array_size, fill_value=np.datetime64("2010-01-01"))
        else:
            return np.zeros(self._sub_array_size, dtype=self._dtype)

    def append(self, value):
        self._cursor += 1
        if self._cursor >= self._sub_array_size:
            self._blocks.append(self._current_block)
            self._current_block = self._create_block()
            self._cursor = 0
        self._current_block[self._cursor] = value

    def collect(self) -> np.ndarray:
        return np.concatenate(self._blocks + [self._current_block[:self._cursor + 1]])


class CdmData:

    def __init__(self):
        self._person_person_id = DynamicArray(np.int64)
        self._person_year_of_birth = DynamicArray(np.int32)
        self._person_month_of_birth = DynamicArray(np.int32)
        self._person_day_of_birth = DynamicArray(np.int32)
        self._person_gender_concept_id = DynamicArray(np.int64)
        self._visit_occurrence_person_id = DynamicArray(np.int64)
        self._visit_occurrence_visit_occurrence_id = DynamicArray(np.int64)
        self._visit_occurrence_visit_start_date = DynamicArray(np.datetime64)
        self._visit_occurrence_visit_end_date = DynamicArray(np.datetime64)
        self._visit_occurrence_visit_concept_id = DynamicArray(np.int64)
        self._condition_occurrence_person_id = DynamicArray(np.int64)
        self._condition_occurrence_visit_occurrence_id = DynamicArray(np.int64)
        self._condition_occurrence_condition_start_date = DynamicArray(np.datetime64)
        self._condition_occurrence_condition_concept_id = DynamicArray(np.int64)
        self._observation_period_person_id = DynamicArray(np.int64)
        self._observation_period_observation_period_id = DynamicArray(np.int64)
        self._observation_period_observation_period_start_date = DynamicArray(np.datetime64)
        self._observation_period_observation_period_end_date = DynamicArray(np.datetime64)

    def add_person(self, person_id: int, year_of_birth: int, month_of_birth: int, day_of_birth: int,
                   gender_concept_id: int):
        self._person_person_id.append(person_id)
        self._person_year_of_birth.append(year_of_birth)
        self._person_month_of_birth.append(month_of_birth)
        self._person_day_of_birth.append(day_of_birth)
        self._person_gender_concept_id.append(gender_concept_id)

    def add_visit_occurrence(self, person_id: int, visit_occurrence_id: int, visit_start_date: np.datetime64,
                             visit_end_date: np.datetime64, visit_concept_id: int):
        self._visit_occurrence_person_id.append(person_id)
        self._visit_occurrence_visit_occurrence_id.append(visit_occurrence_id)
        self._visit_occurrence_visit_start_date.append(visit_start_date)
        self._visit_occurrence_visit_end_date.append(visit_end_date)
        self._visit_occurrence_visit_concept_id.append(visit_concept_id)

    def add_condition_occurrence(self, person_id: int, visit_occurrence_id: int, condition_start_date: np.datetime64,
                                 condition_concept_id: int):
        self._condition_occurrence_person_id.append(person_id)
        self._condition_occurrence_visit_occurrence_id.append(visit_occurrence_id)
        self._condition_occurrence_condition_start_date.append(condition_start_date)
        self._condition_occurrence_condition_concept_id.append(condition_concept_id)

    def add_observation_period(self, person_id: int, observation_period_id: int,
                               observation_period_start_date: np.datetime64,
                               observation_period_end_date: np.datetime64):
        self._observation_period_person_id.append(person_id)
        self._observation_period_observation_period_id.append(observation_period_id)
        self._observation_period_observation_period_start_date.append(observation_period_start_date)
        self._observation_period_observation_period_end_date.append(observation_period_end_date)


class SimulationSettings:
    """
    Settings for the simulation.
    """

    def __init__(self):
        self.person_count = 1000
        self.dynamic_state_count = 10
        self.fixed_state_count = 5
        self.concept_count = 100
        self.visit_multiplier = 2
        self.days_to_simulate = 365 * 2


class Simulation:

    def __init__(self, settings: SimulationSettings):
        self._settings = settings
        self._state_count = settings.dynamic_state_count + settings.fixed_state_count + 2  # + 2 for age and sex
        self._concept_ids = np.arange(settings.concept_count) + 1000000
        # Initialize probabilities and coefficients:
        self._initial_state_probabilities = np.concatenate((
            np.random.uniform(size=settings.dynamic_state_count, high=0.01),
            np.random.uniform(size=settings.fixed_state_count + 2)
        ))
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
        self._cdm_data = CdmData()

    def _simulate_person(self, person_id: int):
        start_date = np.datetime64("2010-01-01") + np.random.randint(0, 365 * 10)
        self._cdm_data.add_observation_period(person_id=person_id,
                                              observation_period_id=person_id,
                                              observation_period_start_date=start_date,
                                              observation_period_end_date=start_date + self._settings.days_to_simulate)
        state = np.random.binomial(n=1, p=self._initial_state_probabilities)

        # state = sparse.csr_matrix(state)
        for t in range(self._settings.days_to_simulate):
            # print(f"Day {t}, state: {state}")
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
                # Role dice again to observe a concept:
                observed_concepts = observed_concepts | np.random.binomial(n=self._settings.visit_multiplier,
                                                                           p=concept_probabilities)
                concept_ids = self._concept_ids[observed_concepts != 0]
                for concept_id in concept_ids:
                    self._cdm_data.add_condition_occurrence(person_id=person_id,
                                                            condition_start_date=start_date + t,
                                                            condition_concept_id=concept_id,
                                                            visit_occurrence_id=1)
                self._cdm_data.add_visit_occurrence(person_id=person_id,
                                                    visit_start_date=start_date + t,
                                                    visit_end_date=start_date + t,
                                                    visit_concept_id=9201,
                                                    visit_occurrence_id=1)
                # print(f"Visit on day {t} with concept IDs: {concept_ids}")

    def simulate(self):
        for i in range(self._settings.person_count):
            self._simulate_person(person_id=i)
            if i % 100 == 0:
                print(f"Simulated {i} persons")


if __name__ == "__main__":
    sim_settings = SimulationSettings()
    simulation = Simulation(sim_settings)
    simulation.simulate()
    # cProfile.runctx("simulation.simulate()", locals={"simulation": simulation}, globals={}, filename="../stats")
