from typing import Optional, List, Tuple, Dict

import numpy as np

from env.map import AbstractMap
from .Path import Path, Coordinate
from .AbstractAgent import AbstractAgent, AgentID
from .conflict_resolution import AbstractPriorityProtocol
from .repairing import AbstractRepairingStrategy
from .utils import find_current_and_goal_points


class Agent(AbstractAgent):
    repairing_strategy: AbstractRepairingStrategy  #: **Repairing Strategy** to fix a other_path when a conflict occurs
    priority_protocol: AbstractPriorityProtocol  #: **Priority Protocol** determines the priority of agents in case of conflict between multiple agents.
    other_agents: List[AbstractAgent]  #: Other agents
    other_agents_path: Optional[Dict[AgentID, Path]]  #: Other agents' path
    goal_position: Optional[Tuple[int, int]]  #: Goal position

    def __init__(self, agent_id: int, t_max: int, repairing_strategy: AbstractRepairingStrategy,
                 priority_protocol: AbstractPriorityProtocol, path: Optional[Path] = None):
        super().__init__(agent_id, t_max, path)

        self.repairing_strategy = repairing_strategy
        self.priority_protocol = priority_protocol
        self.other_agents = []
        self.other_agents_path = None
        self.goal_position = None

    def resolve(self, agents: Optional[List[AbstractAgent]], scenario: AbstractMap, t: int,
                real_t: Optional[int] = None):

        if real_t is None:
            real_t = t

        if len(agents) == 0:  # Conflict with an obstacle
            other_paths = list(self.other_agents_path.values())

            fixed_path = self.repairing_strategy.repair(self.path, self, t, real_t, scenario, other_paths)

            if fixed_path is None:
                return

            # If the repairing approach cannot solve, do nothing
            if self.goal_position is None:
                _, (goal_i, goal_j) = find_current_and_goal_points(scenario.get_data(), self.id)
                self.goal_position = (goal_i, goal_j)

            if len(fixed_path) == 0 or fixed_path[-1][0] != self.goal_position[0] or \
                    fixed_path[-1][1] != self.goal_position[1]:
                return

            self.path = fixed_path
        else:  # Conflict with agent(s)
            self.priority_protocol.resolve(agents, scenario, t, real_t)

    def detect_conflict_with_other_agents(self, t: int) -> list:
        """
            This method detect any conflicts (i.e., possible collision and swapping) with other agents

            :param t: Current time step
            :return: List of conflicting agents
        """

        next_i, next_j = self.path[t + 1]
        current_i, current_j = self.path[t]

        # Check for conflict with other agents
        conflict_other_agents = []

        for agent in self.other_agents:
            if len(agent.path) > t + 1:
                coordinate = agent.path[t + 1]

                if next_i == coordinate[0] and next_j == coordinate[1]:
                    conflict_other_agents.append(agent)

                previous_coordinate = agent.path[t]

                if previous_coordinate[0] == next_i and previous_coordinate[1] == next_j \
                        and coordinate[0] == current_i and coordinate[1] == current_j:  # Swapping

                    conflict_other_agents.append(agent)

        if len(conflict_other_agents) > 0:  # Conflict with another agent
            conflict_other_agents.append(self)

        return conflict_other_agents

    def exchange_information(self, scenario: AbstractMap, t: int):
        if self.other_agents_path is None:
            self.other_agents_path = {}

            for other_agent in self.other_agents:
                self.other_agents_path[other_agent.id] = other_agent.path.copy()

            return

        current_i, current_j = self.path[t]

        communicated_agents = set()

        for i in range(max(0, current_i - 2), min(scenario.n, current_i + 3)):
            for j in range(max(0, current_j - 2), min(scenario.n, current_j + 3)):
                if scenario[i, j, 1] > 0:
                    communicated_agents.add(int(scenario[i, j, 1]))

        if len(communicated_agents) == 0:
            return

        for other_agent in self.other_agents:
            if other_agent.id in communicated_agents:
                self.other_agents_path[other_agent.id] = other_agent.path.copy()


    def update(self, scenario: AbstractMap, t: int):
        if t >= len(self.path) - 1:
            return

        map_data = scenario.get_data()

        self.exchange_information(scenario, t)

        next_i, next_j = self.path[t + 1]

        if map_data[next_i, next_j, 0] != 0:  # Conflict with an obstacle
            clone_map = scenario.clone()
            clone_map.set_data(map_data)

            self.resolve([], clone_map, t)

            return

        # Conflict with other agents
        conflict_agents = self.detect_conflict_with_other_agents(t)
        if len(conflict_agents) > 0:
            clone_map = scenario.clone()
            clone_map.set_data(map_data)

            self.resolve(conflict_agents, clone_map, t)

            return

        for current_t in range(t, len(self.path) - 1):
            current_i, current_j = self.path[current_t]
            next_i, next_j = self.path[current_t + 1]

            if map_data[next_i, next_j, 0] == 1:  # Conflict with a static obstacle
                clone_map = scenario.clone()
                clone_map.set_data(map_data)

                self.resolve([], clone_map, current_t, t)

                return

            if map_data[next_i, next_j, 0] >= 2 and abs(t - current_t) <= 2:  # Conflict with a dynamic obstacle
                clone_map = scenario.clone()
                clone_map.set_data(map_data)

                self.resolve([], clone_map, current_t, t)

                return

            # Conflict with other agents
            conflict_agents = self.detect_conflict_with_other_agents(current_t)
            if len(conflict_agents) > 0:
                clone_map = scenario.clone()
                clone_map.set_data(map_data)

                self.resolve(conflict_agents, clone_map, current_t, t)

                return

            # Update scenario
            map_data[current_i, current_j, 1] = 0
            map_data[next_i, next_j, 1] = self.id
