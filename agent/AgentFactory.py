import time
from typing import List, Dict
import numpy as np
from .AbstractAgent import AgentID
from .Agent import Agent
from .LPAgent import LPAgent
from env.map import AbstractMap
from .conflict_resolution import AbstractPriorityProtocol
from .initial_path.InitialPathStrategy import AbstractInitialPathStrategy
from .repairing import AbstractRepairingStrategy
from .utils import find_current_and_goal_points


class AgentFactory:
    @staticmethod
    def generate(map: AbstractMap, initial_path_strategy: AbstractInitialPathStrategy,
                 repairing_strategy: AbstractRepairingStrategy, priority_protocol: AbstractPriorityProtocol,
                 t_max_multiplier: float = 2., **kwargs) \
            -> (List[Agent], float):
        """
            Generate the agents with initial paths

            :param map: Current scenario
            :param initial_path_strategy: Initial other_path strategy
            :param repairing_strategy: Repairing strategy
            :param priority_protocol: Priority protocol
            :param t_max_multiplier: Time limitation multiplier
            :param kwargs: Additional parameters for initial other_path strategy
            :return: List of agents and the total elapsed time for initial other_path strategy
        """

        number_of_agents = np.max(map.get_data()[:, :, 1])

        agent_ids = [agent_id for agent_id in range(1, number_of_agents + 1)]

        # Determine t_max values for each agent
        t_max: Dict[AgentID, int] = {}

        for agent_id in agent_ids:
            (start_i, start_j), (goal_i, goal_j) = find_current_and_goal_points(map.get_data(), agent_id)

            distance = abs(start_i - goal_i) + abs(start_j - goal_j)

            t_max[agent_id] = round(distance * t_max_multiplier)

        # Generate initial paths
        start_time = time.time()
        paths = initial_path_strategy.generate_initial_paths(map, agent_ids, t_max, **kwargs)
        end_time = time.time()

        # Set Repairing Strategy to Priority Protocol
        priority_protocol.repairing_strategy = repairing_strategy

        # Create agents
        agents = [Agent(agent_id, t_max[agent_id], repairing_strategy, priority_protocol, paths[agent_id])
                  for agent_id in agent_ids]

        # Set other agents
        for agent in agents:
            agent.other_agents = [other_agent for other_agent in agents if other_agent.id != agent.id]

        return agents, end_time - start_time

    @staticmethod
    def generate_for_lp(map: AbstractMap, initial_path_strategy: AbstractInitialPathStrategy,
                        repairing_strategy: AbstractRepairingStrategy, priority_protocol: AbstractPriorityProtocol,
                        t_max_multiplier: float = 2., **kwargs) \
            -> (List[Agent], float):
        """
            Generate the agents with initial paths for LP Agent

            :param map: Current scenario
            :param initial_path_strategy: Initial other_path strategy
            :param repairing_strategy: Repairing strategy
            :param priority_protocol: Priority protocol
            :param t_max_multiplier: Time limitation multiplier
            :param kwargs: Additional parameters for initial other_path strategy
            :return: List of agents and the total elapsed time for initial other_path strategy
        """

        number_of_agents = np.max(map.get_data()[:, :, 1])

        agent_ids = [agent_id for agent_id in range(1, number_of_agents + 1)]

        # Determine t_max values for each agent
        t_max: Dict[AgentID, int] = {}

        for agent_id in agent_ids:
            (start_i, start_j), (goal_i, goal_j) = find_current_and_goal_points(map.get_data(), agent_id)

            distance = abs(start_i - goal_i) + abs(start_j - goal_j)

            t_max[agent_id] = round(distance * t_max_multiplier)

        # Generate initial paths
        start_time = time.time()
        paths = initial_path_strategy.generate_initial_paths(map, agent_ids, t_max, **kwargs)
        end_time = time.time()

        # Set Repairing Strategy to Priority Protocol
        priority_protocol.repairing_strategy = repairing_strategy

        # Create agents
        agents = [LPAgent(agent_id, t_max[agent_id], repairing_strategy, priority_protocol, paths[agent_id])
                  for agent_id in agent_ids]

        # Set other agents
        for agent in agents:
            agent.other_agents = [other_agent for other_agent in agents if other_agent.id != agent.id]

        return agents, end_time - start_time
