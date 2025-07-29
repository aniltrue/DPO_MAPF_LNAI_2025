import math
from typing import Optional, List, Dict
import numba
import numpy as np
from env.map import AbstractMap
from .RepairingStrategy import AbstractRepairingStrategy
from agent.Path import Path, Coordinate
from agent.AbstractAgent import AgentID, AbstractAgent
from agent.initial_path.aco import optimize, get_heuristic_matrix


class ACOInformedWAPRepairingStrategy(AbstractRepairingStrategy):
    """
        This repairing approach employs *ACO* algorithm to resolve the conflicts by utilizing the previous pheromones.
    """

    INITIAL_BETA: float                              #: Initial beta value for dynamic beta adjustment
    FINAL_BETA: float                                #: Final beta value for dynamic beta adjustment
    MAX_ITER: int                                    #: Max. Iteration for repairing
    NUMBER_OF_ANTS: int                              #: Number of ants
    INFORMATION_MULTIPLIER: float                    #: Information Multiplier for Pheromone Initiation
    pheromones: Optional[Dict[AgentID, np.ndarray]]  #: Pheromone matrix for each agent

    def __init__(self, max_iter: int = 150, initial_b: float = 5.0, end_b: float = 0.5, number_of_ants: int = 75,
                 information_multiplier: float = 3.0):
        super().__init__()

        self.MAX_ITER = max_iter
        self.INITIAL_BETA = initial_b
        self.FINAL_BETA = end_b
        self.NUMBER_OF_ANTS = number_of_ants
        self.INFORMATION_MULTIPLIER = information_multiplier
        self.pheromones = {}

    @property
    def name(self) -> str:
        return "ACO-Based Informed WAP Repairing Strategy"

    def repair(self, path: Path, agent: AbstractAgent, t: int, real_t: int, scenario: AbstractMap,
               other_paths: Optional[List[Path]]) -> Optional[Path]:

        if t < real_t:
            return None

        # Waiting strategy
        path_with_waiting = self.waiting_strategy(agent, t, real_t, scenario)

        if path_with_waiting is not None:
            if 1 < t < len(path):
                return path_with_waiting

        agent_id = agent.id

        # Collect other best paths
        other_best_paths: List[Path] = []

        for other_path in other_paths:
            if len(other_path) > t:
                other_best_paths.append(other_path[t:])

        # Get heuristic information
        heuristics = get_heuristic_matrix(scenario.get_data(), agent_id, agent.t_max)

        # Initiate variables
        best_path = []
        best_cost = agent.t_max + 1 - t

        # Initiate scenario data
        map_data = scenario.get_data()

        # Get positions
        self.find_positions(map_data, [agent_id])

        # Initiate beta parameter for dynamic adjustment
        beta = self.INITIAL_BETA
        beta_decay = (beta - self.FINAL_BETA) / self.MAX_ITER

        # Initiate Pheromones
        pheromones = np.ones((scenario.n, scenario.n))

        # Agent position
        position = (agent.path[t], self.positions[agent_id][1])

        # Previous path after conflict
        if len(agent.path) > t + 1:
            candidate_path = agent.path[t + 1:]

            previous_path = []

            for candidate_path_index in range(len(candidate_path) - 1, -1, -1):
                candidate_position = candidate_path[candidate_path_index]
                candidate_path_t = candidate_path_index + 1

                if map_data[candidate_position[0], candidate_position[1], 0] != 0:
                    break

                validity = True
                for other_best_path in other_best_paths:
                    if len(other_best_path) <= candidate_path_t:
                        continue

                    if other_best_path[candidate_path_t][0] == candidate_position[0]\
                            and other_best_path[candidate_path_t][1] == candidate_position[1]:
                        validity = False
                        break

                if not validity:
                    break
                else:
                    previous_path.insert(0, candidate_position)

            if len(previous_path) == 0:
                previous_path = None
        else:
            previous_path = None

        # Early stopping threshold
        non_improve_threshold = math.ceil(self.MAX_ITER / 3)
        non_improve_counter = 0

        for i in range(self.MAX_ITER):  # Start iteration
            # Run ACO
            if len(other_best_paths) > 0:
                cost, aco_path, updated_pheromone = optimize(map_data, pheromones, heuristics, agent.t_max - t,
                                                             numba.typed.List(other_best_paths), position, previous_path,
                                                             beta=beta, alpha=1.0, number_of_ants=self.NUMBER_OF_ANTS,
                                                             weighted_average=True)
            else:
                cost, aco_path, updated_pheromone = optimize(map_data, pheromones, heuristics, agent.t_max - t, None,
                                                             position, previous_path, beta=beta, alpha=1.0,
                                                             number_of_ants=self.NUMBER_OF_ANTS, weighted_average=True)

            if cost < best_cost:  # If the new other_path is better
                best_cost = cost
                best_path = aco_path
            elif len(best_path) > 1:
                non_improve_counter += 1

            if non_improve_counter > non_improve_threshold:
                break

            # Update pheromone
            pheromones = updated_pheromone

            # Decrease beta value
            beta -= beta_decay
            beta = max(beta, self.FINAL_BETA)

        # Check for previous other_path
        if len(best_path) <= 1:
            return self.repair(path, agent, t - 1, real_t, scenario, other_paths)

        self.pheromones[agent.id] = pheromones

        return path[:t] + best_path
