from abc import ABC, abstractmethod
from typing import List, Dict, Tuple

import numpy as np

from agent.AbstractAgent import AgentID
from agent.utils import find_current_and_goal_points
from env.map import AbstractMap
from agent.Path import Path, Coordinate


class AbstractInitialPathStrategy(ABC):
    """
        **Initial Path Strategies** generates an initial other_path for each agent in a *centralized and offline* manner.
    """

    positions: Dict[AgentID, Tuple[Coordinate, Coordinate]]   #: Positions for each agent

    def __init__(self):
        self.positions = {}

    @property
    @abstractmethod
    def name(self) -> str:
        """
            Name of the *Initial Path Strategy* for logging purposes.

            :return: The name of the strategy
        """

        ...

    def find_positions(self, map_data: np.ndarray, agent_ids):
        """
            This method finds and remembers all starting and goal positions of each agent

            :param map_data: Map data
            :param agent_ids: List of agent ids
        """
        for agent_id in agent_ids:
            if agent_id not in self.positions:
                self.positions[agent_id] = find_current_and_goal_points(map_data, agent_id)

    @abstractmethod
    def generate_initial_paths(self, scenario: AbstractMap, agent_ids: List[AgentID], t_max: Dict[AgentID, int],
                               **kwargs) -> Dict[AgentID, Path]:
        """
            This method generates initial other_path for each agent id.

            :param scenario: Current Map
            :param agent_ids: List of agent ids
            :param t_max: Maximum time-step for each agent id.
            :return: Agent ID -> Path dictionary
        """

        ...
