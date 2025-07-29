from __future__ import annotations
import pickle
from abc import ABC, abstractmethod
from typing import Union, Tuple, Dict, Optional
import numpy as np


class AbstractMap(ABC):
    _data: np.ndarray   #: Map data
    n: int              #: Map size (n x n)

    def __init__(self, n: int):
        """
            Constructor

            :param n: Map Size (n x n)
        """

        self.n = n

    def initiate(self):
        """
            This method is called at the beginning of the scenario.
        """

        pass

    @abstractmethod
    def step(self, **kwargs):
        """
             This method updates the current state (i.e., scenario state)
        """

    @abstractmethod
    def check_collision(self, **kwargs) -> bool:
        """
            This method checks whether any collision is detected, or not.

            :return: Whether any collision is detected, or not.
        """

    def check_data(self) -> bool:
        """
            This method controls whether the scenario data is proper, or not

            :return: Whether the scenario data is proper, or not
        """

        return isinstance(self._data, np.ndarray) and self._data.shape[0] == self._data.shape[1]

    def get_data(self) -> np.ndarray:
        """
            This method provides the scenario data as *NumPy* array

            :return: Map data
        """

        return self._data.copy()

    def get_raw_data(self) -> np.ndarray:
        """
            This method provides the **raw** scenario data without any modification as *NumPy* array

            :return: Raw scenario data
        """

        return self._data.copy()

    @abstractmethod
    def get_lp_data(self) -> Dict[str, Union[int, np.ndarray]]:
        """
            This method converts the scenario data into the time-based data for LP.

            :return: LP Parameters
        """

        ...

    def set_data(self, map_data: np.ndarray):
        """
            This method changes the scenario data as *NumPy* array

            :param map_data: Map data as *NumPy* array
        """

        self._data = map_data

    def __getitem__(self, indices: Union[Tuple[int, int], Tuple[int, int, int]]) \
            -> Union[np.ndarray, int]:

        return self._data[indices]

    def __setitem__(self, key: Union[Tuple[int, int], Tuple[int, int, int]], value: int):
        self._data[key] = value

    def save(self, file_path: str):
        """
            This method saves the scenario data as *Pickle* file format.

            :param file_path: File other_path to save
        """

        with open(file_path, "wb") as f:
            pickle.dump(self._data, f)

    def load(self, file_path: str):
        """
            This method load the scenario data from the given *Pickle* file format

            :param file_path: File other_path to load
        """

        with open(file_path, "rb") as f:
            self.set_data(pickle.load(f))

        assert self.check_data(), "Invalid scenario data."

        self.n = self._data.shape[0]

    @abstractmethod
    def clone(self) -> AbstractMap:
        """
            This method generates a copy of this scenario.

            :return: Clone scenario object
        """
        ...

    @staticmethod
    @abstractmethod
    def load_map_factory(file_path: str) -> AbstractMap:
        """
            This method load the scenario data from the given *Pickle* file format

            :param file_path: File other_path to load
            :return: Loaded scenario
        """

        ...

    @staticmethod
    @abstractmethod
    def random_map_factory(n: int, number_of_agents: int, density: float, seed: Optional[int],
                           **kwargs) -> AbstractMap:
        """
            This method randomly generates maps based on given parameters.

            :param n: Map size (n x n)
            :param number_of_agents: Number of agents
            :param density: Static obstacle density
            :param seed: Random seed. If it is given as *None*, no seed is defined for random object.
            :return: Generated Map
        """

        ...
