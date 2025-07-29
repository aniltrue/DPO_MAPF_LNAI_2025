import random
from typing import Optional, Tuple, Set, List

from .Map import AbstractMap
from .DPOMap import DPOMap



class BenchmarkMapWrapper:
    map_path: str                   #: Path to the map file
    static_map: DPOMap              #: Map object with only static obstacles
    nodes: Set[Tuple[int, int]]     #: The set of used nodes to avoid duplicates

    def __init__(self, map_path: str):
        """
            Constructor read the map file to initiate static obstacles

        :param map_path: Path to the map file
        """
        self.map_path = map_path
        self.nodes = set()

        # Read the map file
        with open(self.map_path, 'r') as f:
            lines = f.readlines()

            # Get size of map
            width = int(lines[1].split()[1])
            height = int(lines[2].split()[1])

            # Map must be square
            n = max(width, height)

            # Generate static obstacles
            self.static_map = DPOMap(n)

            for i, line in enumerate(lines[4:]):
                if line.strip() == '':
                    break

                values = line.strip()

                for j, value in enumerate(values):
                    if value != '.':
                        self.static_map[i, j, 0] = 1
                        self.nodes.add((i, j))

            # Fill out with statics
            while width < n:
                for i in range(n):
                    self.static_map[i, width, 0] = 1
                    self.nodes.add((i, width))

                width += 1

            while height < n:
                for j in range(n):
                    self.static_map[height, j, 0] = 1
                    self.nodes.add((height, j))

                height += 1

    def __call__(self, scenario_path: str, seed: int, number_of_dynamic_obstacles: Optional[int] = None) -> (AbstractMap, int, int):
        """
            This method provides a map object based on given scenario

        :param scenario_path: Path to the scenario file
        :param seed: Random seed to generate dynamic obstacles
        :param number_of_dynamic_obstacles: Number of dynamic obstacles, optional.
        :return: Generated map, number of agents and target subset id
        """
        # Initiate variables
        rnd = random.Random(seed)

        nodes = self.nodes.copy()
        map_obj = self.static_map.clone()

        # Select a target subset
        subsets = self.extract_subset_ids(scenario_path)

        target_subset = rnd.choice(subsets)

        number_of_agents = self.read_scenario(map_obj, nodes, scenario_path, target_subset=target_subset)

        DPOMap.random_dynamic_obstacles(map_obj,
                                        nodes,
                                        number_of_agents,
                                        random.Random(seed),
                                        number_of_dynamic_obstacles if not number_of_dynamic_obstacles else number_of_agents // 4)

        return map_obj, number_of_agents, target_subset

    @staticmethod
    def read_scenario(map_obj: DPOMap, nodes: Set[Tuple[int, int]], scenario_path: str, target_subset: int = 0) -> int:
        """
            This method read the given scenario file and return the number of agents

        :param map_obj: Current Map object
        :param nodes: Set of used nodes
        :param scenario_path: Path to the scenario file
        :param target_subset: The subset id
        :return: Number of agents in that scenario
        """
        number_of_agents = 0

        with open(scenario_path, 'r') as f:
            lines = f.readlines()

            for line in lines[1:]:
                # Read the line
                values = line.strip().split()

                # Validate subset
                subset_id = int(values[0])

                if subset_id != target_subset:
                    continue

                number_of_agents += 1

                # Get coordinates
                start_j, start_i = int(values[4]), int(values[5])
                goal_j, goal_i = int(values[6]), int(values[7])

                # Update map_obj
                map_obj[start_i, start_j, 1] = number_of_agents
                map_obj[goal_i, goal_j, 2] = number_of_agents

                # Append to nodes set
                nodes.add((start_i, start_j))
                nodes.add((goal_i, goal_j))

        return number_of_agents


    @staticmethod
    def extract_subset_ids(scenario_path: str) -> List[int]:
        """
            This method read the scenario file and return the list of subset ids

        :param scenario_path: Path to the scenario file
        :return: List of subset ids
        """
        subsets = set()

        with open(scenario_path, 'r') as f:
            lines = f.readlines()

            for line in lines[1:]:
                # Read the line
                values = line.strip().split()

                # Extract the subset id
                subset_id = int(values[0])

                if subset_id not in subsets:
                    subsets.add(subset_id)

        return list(subsets)
