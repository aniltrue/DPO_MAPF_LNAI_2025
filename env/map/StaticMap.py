import random
from typing import Union, Dict, Optional
import numpy as np
from env.map.Map import AbstractMap
from env.utils import check_for_collision


class StaticMap(AbstractMap):
    """
        - **Static Obstacle**: True
        - **Dynamic Obstacle**: False
        - **Partial Observability**: False
    """

    def __init__(self, n: int):
        super().__init__(n)

        if n != -1:
            self._data = np.zeros((n, n, 3), dtype=np.int32)

    def step(self, **kwargs):
        pass

    def check_collision(self, **kwargs) -> bool:
        return check_for_collision(self._data, [0])

    def clone(self) -> AbstractMap:
        map_obj = StaticMap(-1)

        map_obj.n = self.n
        map_obj._data = self._data.copy()

        return map_obj

    def get_lp_data(self) -> Dict[str, Union[int, np.ndarray]]:
        n = self._data.shape[0]
        number_of_agents = np.max(self._data[:, :, 1])

        # Max. Time-Step for the environment
        T = n * 4

        # Obstacles
        obstacles = np.stack([self._data[:, :, 0] for _ in range(T)], axis=-1)

        # Starting & goal matrix
        starting_matrix = np.stack(
            [np.zeros_like(self._data[:, :, 1])] + [self._data[:, :, 1] == m
                                                    for m in range(1, np.max(self._data[:, :, 1]) + 1)], axis=-1,
            dtype=np.float32)

        goal_matrix = np.stack(
            [np.zeros_like(self._data[:, :, 2])] + [self._data[:, :, 2] == m
                                                    for m in range(1, np.max(self._data[:, :, 2]) + 1)], axis=-1,
            dtype=np.float32)

        return {
            "n": n,
            "s": starting_matrix,
            "g": goal_matrix,
            "o": obstacles,
            "T": T,
            "m": number_of_agents
        }


    @staticmethod
    def load_map_factory(file_path: str) -> AbstractMap:
        map_obj = StaticMap(-1)

        map_obj.load(file_path)

        return map_obj

    @staticmethod
    def random_map_factory(n: int, number_of_agents: int, density: float, seed: Optional[int],
                           **kwargs) -> AbstractMap:

        assert n > 0, "Map size must be positive"
        assert number_of_agents > 0, "Number of agents must be positive"
        assert 0 <= density <= 1, "Density must be in range [0, 1]"

        rnd = random.Random() if seed is None else random.Random(seed)

        map_obj = StaticMap(n)

        nodes = set()

        # Generate Agents
        min_distance = n // 2

        for k in range(1, number_of_agents + 1):
            # Starting point
            i = rnd.randint(0, n - 1)
            j = rnd.randint(0, n - 1)

            while (i, j) in nodes:
                i = rnd.randint(0, n - 1)
                j = rnd.randint(0, n - 1)

            map_obj[i, j, 1] = k
            nodes.add((i, j))

            start_i = i
            start_j = j

            # Goal point
            i = rnd.randint(0, n - 1)
            j = rnd.randint(0, n - 1)

            while (i, j) in nodes or abs(i - start_i) + abs(j - start_j) < min_distance:
                i = rnd.randint(0, n - 1)
                j = rnd.randint(0, n - 1)

            map_obj[i, j, 2] = k

        # Generate obstacles
        number_of_obstacle = round(n * n * density)

        for k in range(number_of_obstacle):
            i = rnd.randint(0, n - 1)
            j = rnd.randint(0, n - 1)

            while (i, j) in nodes:
                i = rnd.randint(0, n - 1)
                j = rnd.randint(0, n - 1)

            map_obj[i, j, 0] = 1
            nodes.add((i, j))

        return map_obj
