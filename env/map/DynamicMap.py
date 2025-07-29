from typing import Union, Dict, Tuple, Set, Optional
import random
import numpy as np
from env.map.Map import AbstractMap
from env.utils import check_for_collision, move_dynamic_obstacles


class DynamicMap(AbstractMap):
    """
        - **Static Obstacle**: True
        - **Dynamic Obstacle**: True
        - **Partial Observability**: False
    """

    def __init__(self, n: int):
        super().__init__(n)

        if n != -1:
            self._data = np.zeros((n, n, 4), dtype=np.int32)

    def step(self, **kwargs):
        """
            This method plays *dynamic obstacles*
        """

        move_dynamic_obstacles(self._data)

    def check_collision(self, **kwargs) -> bool:
        return check_for_collision(self._data, [0, 3])

    def check_data(self) -> bool:
        return super().check_data() and self._data.shape[2] == 4

    def clone(self) -> AbstractMap:
        map_obj = DynamicMap(-1)

        map_obj.n = self.n
        map_obj._data = self._data.copy()

        return map_obj

    def get_data(self) -> np.ndarray:
        """
            Convert *Dynamic Obstacles* to *Static Obstacles*

            :return: Converted Map Data
        """
        data_clone = self._data.copy()
        data_clone_result = self._data.copy()

        data_clone_result[:, :, 0] += (1 - data_clone[:, :, 0]) * (data_clone[:, :, 3] > 0) * 2

        move_dynamic_obstacles(data_clone)
        data_clone_result[:, :, 0] += (1 - data_clone[:, :, 0]) * (data_clone[:, :, 3] > 0) * 2

        return data_clone_result

    def get_lp_data(self) -> Dict[str, Union[int, np.ndarray]]:
        n = self._data.shape[0]
        number_of_agents = np.max(self._data[:, :, 1])

        # Max. Time-Step for the environment
        T = n * 4

        # Obstacles
        obstacles = np.stack([self._data[:, :, 0] for _ in range(T)], axis=-1)

        clone_map = np.copy(self._data)

        for t in range(0, T):
            obstacles[:, :, t] += clone_map[:, :, 3] >= 1
            move_dynamic_obstacles(clone_map)

        obstacles = np.minimum(1, obstacles)

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
        map_obj = DynamicMap(-1)

        map_obj.load(file_path)

        return map_obj

    @staticmethod
    def random_map_factory(n: int, number_of_agents: int, density: float, seed: Optional[int],
                           **kwargs) -> AbstractMap:

        assert n > 0, "Map size must be positive"
        assert number_of_agents > 0, "Number of agents must be positive"
        assert 0 <= density <= 1, "Density must be in range [0, 1]"

        rnd = random.Random() if seed is None else random.Random(seed)

        map_obj = DynamicMap(n)

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

        # Generate static obstacles
        number_of_obstacle = round(n * n * density)

        for k in range(number_of_obstacle):
            i = rnd.randint(0, n - 1)
            j = rnd.randint(0, n - 1)

            while (i, j) in nodes:
                i = rnd.randint(0, n - 1)
                j = rnd.randint(0, n - 1)

            map_obj[i, j, 0] = 1
            nodes.add((i, j))

        # Generate dynamic obstacles
        number_of_dynamic_obstacles = kwargs.get("number_of_dynamic_obstacles", None)
        DynamicMap.random_dynamic_obstacles(map_obj, nodes, number_of_agents, rnd, number_of_dynamic_obstacles)

        return map_obj

    @staticmethod
    def random_dynamic_obstacles(map_obj: AbstractMap,
                                 nodes: Set[Tuple[int, int]],
                                 number_of_agents: int,
                                 rnd: random.Random,
                                 number_of_dynamic_obstacles: Optional[int] = None):
        """
            This method randomly generates dynamic obstacles

        :param map_obj: Static map
        :param nodes: The set of used nodes
        :param number_of_agents: Number of agents
        :param rnd: Random object
        :param number_of_dynamic_obstacles: Number of dynamic obstacles.
        If it is not defined, it will equal to the number of agents
        """
        n = map_obj.n

        # Define the number of dynamic obstacles
        if number_of_dynamic_obstacles is not None:
            assert number_of_dynamic_obstacles > 0, "Number of dynamic obstacles must be positive"

            number_of_obstacle = number_of_dynamic_obstacles
        else:
            number_of_obstacle = number_of_agents

        # Randomly add dynamic obstacles
        for k in range(number_of_obstacle):
            i = rnd.randint(0, n - 1)
            j = rnd.randint(0, n - 1)

            while (i, j) in nodes:
                i = rnd.randint(0, n - 1)
                j = rnd.randint(0, n - 1)

            move = rnd.randint(1, 4)
            map_obj[i, j, 3] = move  # Set random move
            nodes.add((i, j))
