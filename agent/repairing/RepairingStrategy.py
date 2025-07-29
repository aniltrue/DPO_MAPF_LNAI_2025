from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Tuple
import numpy as np
from agent.AbstractAgent import AbstractAgent, AgentID
from agent.Path import Path, Coordinate
from agent.utils import find_current_and_goal_points
from env.map.Map import AbstractMap


class AbstractRepairingStrategy(ABC):
    """
        **Repairing Strategies** fix the other_path of an agent in a *decentralized and online* manner if the agent detects a
        possible collision.
    """

    positions: Dict[AgentID, Tuple[Coordinate, Coordinate]]  #: Positions for each agent

    def __init__(self):
        self.positions = {}

    @property
    @abstractmethod
    def name(self) -> str:
        """
            Name of the *Repairing Strategy* for logging purposes.

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
    def repair(self, path: Path, agent: AbstractAgent, t: int, real_t: int, map: AbstractMap,
               other_paths: Optional[List[Path]]) -> Optional[Path]:
        """
            This method provides the repaired other_path for the agent based on the implemented strategy.

            **Note**: This method returns *None* if the other_path cannot be repaired.

            :param path: Current other_path
            :param agent: The target agent
            :param t: Current time-step
            :param real_t: Real current time-step
            :param map: Current scenario
            :param other_paths: The paths of the other agents if the conflict is caused by other agents
            :return: Repaired other_path or None
        """

        ...

    @staticmethod
    def waiting_strategy(agent, t: int, real_t: int, map: AbstractMap) -> Optional[Path]:
        """
            This *waiting* strategy tries **WAIT** action to resolve the conflict.

            :param agent: The target agent
            :param t: Target time-step
            :param real_t: Real current time-step
            :param map: Current scenario
            :return: The new other_path if it is feasible
        """

        if t <= 0 or t < real_t:
            return None

        # New Path with waiting
        new_path = agent.path.copy()
        new_path.insert(t, new_path[t - 1])

        # Check for feasibility
        map_data = map.get_data()

        next_i, next_j = new_path[t + 1]

        if map_data[next_i, next_j, 1] not in [0, agent.id]:  # Conflict with another agent
            return None

        if map_data[next_i, next_j, 0] == 1:  # Conflict with a static obstacle
            return None

        if map_data[next_i, next_j, 0] == 2 and abs(t - real_t) < 2:  # Conflict with a dynamic obstacle
            return None

        for current_t in range(t, len(new_path) - 1):
            current_i, current_j = new_path[current_t]
            next_i, next_j = new_path[current_t + 1]

            # Check for conflict with other agents
            for agent in agent.other_agents:
                if len(agent.path) > current_t + 1:
                    coordinate = agent.path[current_t + 1]

                    # Collision
                    if next_i == coordinate[0] and next_j == coordinate[1]:
                        return None

                    # Swapping
                    previous_coordinate = agent.path[current_t]

                    if previous_coordinate[0] == next_i and previous_coordinate[1] == next_j \
                            and coordinate[0] == current_i and coordinate[1] == current_j:  # Swapping

                        return None

            if map_data[next_i, next_j, 0] == 1:  # Conflict with a static obstacle
                return None

            if map_data[next_i, next_j, 0] == 2 and abs(current_t - real_t) < 2: # Conflict with a dynamic obstacle
                return None

            # Update scenario
            map_data[current_i, current_j, 1] = 0
            map_data[next_i, next_j, 1] = agent.id

        if len(new_path) < agent.t_max + 1:  # Check for capacity
            return new_path
        else:
            return None
