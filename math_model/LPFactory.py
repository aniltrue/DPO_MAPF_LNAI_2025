from typing import List, Dict
import numpy as np

from agent import Path
from agent.AbstractAgent import AgentID
from env.map import AbstractMap
from agent.utils import find_current_and_goal_points
from .math_model import solve


class LPFactory:
    @staticmethod
    def determine_time_limitations(map: AbstractMap, t_max_multiplier: float = 2.) -> Dict[AgentID, int]:
        """
            This method determines the maximum time-step limitations for each agent

            :param map: Current scenario
            :param t_max_multiplier: Limitations depends on the *manhattan distance* and the given multiplier.
            :return: Maximum time-step limitations for each agent
        """
        number_of_agents = np.max(map.get_data()[:, :, 1])

        agent_ids = [agent_id for agent_id in range(1, number_of_agents + 1)]

        # Determine t_max values for each agent
        t_max: Dict[AgentID, int] = {}

        for agent_id in agent_ids:
            (start_i, start_j), (goal_i, goal_j) = find_current_and_goal_points(map.get_data(), agent_id)

            distance = abs(start_i - goal_i) + abs(start_j - goal_j)

            t_max[agent_id] = round(distance * t_max_multiplier)

        return t_max

    @staticmethod
    def solve_for_objective(map: AbstractMap, t_max_multiplier: float = 2., **kwargs) -> (bool, int, float):
        """
            This method solves the given scenario via Gurobi

            :param map: Given scenario
            :param t_max_multiplier: Time limitation multiplier
            :param kwargs: Gurobi parameters
            :return: Feasibility, objective value and MIP gap

        """
        # Prepare parameters
        t_max = LPFactory.determine_time_limitations(map, t_max_multiplier)

        parameters = map.get_lp_data()

        # Solve
        model, _, feasibility = solve(parameters, t_max, **kwargs)

        if not feasibility:
            cost = sum(t_max.values())
            mip_gap = -1
        else:
            cost = model.getObjective().getValue()
            mip_gap = model.MIPGap

        return feasibility, cost, mip_gap

    @staticmethod
    def solve(map: AbstractMap, t_max_multiplier: float = 2., **kwargs) -> Dict[AgentID, Path]:
        """
            This method solves the given scenario via Gurobi

            :param map: Given scenario
            :param t_max_multiplier: Time limitation multiplier
            :param kwargs: Gurobi parameters
            :return: Path for agents
        """
        # Prepare parameters
        t_max = LPFactory.determine_time_limitations(map, t_max_multiplier)

        parameters = map.get_lp_data()

        # Solve
        model, paths, feasibility = solve(parameters, t_max, presolve=False, **kwargs)

        return paths
