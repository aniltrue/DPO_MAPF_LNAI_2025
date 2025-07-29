import math
from typing import Optional, List, Dict
import numba
import numpy as np
from env.map import AbstractMap
from .RepairingStrategy import AbstractRepairingStrategy
from agent.Path import Path, Coordinate
from agent.AbstractAgent import AgentID, AbstractAgent
from agent.initial_path.aco import optimize, get_heuristic_matrix


class OnlyACORepairingStrategy(AbstractRepairingStrategy):
    """
        This repairing approach employs classical *ACO* algorithm to resolve the conflicts.
    """

    MAX_ITER: int                                    #: Max. Iteration for repairing
    NUMBER_OF_ANTS: int                              #: Number of ants
    pheromones: Optional[Dict[AgentID, np.ndarray]]  #: Pheromone matrix for each agent

    def __init__(self, max_iter: int = 150, number_of_ants: int = 75):
        super().__init__()

        self.MAX_ITER = max_iter
        self.NUMBER_OF_ANTS = number_of_ants
        self.pheromones = {}

    @property
    def name(self) -> str:
        return "Only ACO Repairing Strategy"

    def repair(self, path: Path, agent: AbstractAgent, t: int, real_t: int, scenario: AbstractMap,
               other_paths: Optional[List[Path]]) -> Optional[Path]:

        if t < real_t:
            return None

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

        # Initiate Pheromones
        pheromones = np.ones((scenario.n, scenario.n))

        # Agent position
        position = (agent.path[t], self.positions[agent_id][1])

        # Max. Move for Ants
        ant_limit = map_data.shape[0] + map_data.shape[1]

        for i in range(self.MAX_ITER):  # Start iteration
            # Run ACO
            if len(other_best_paths) > 0:
                cost, aco_path, updated_pheromone = optimize(map_data, pheromones, heuristics, ant_limit,
                                                             numba.typed.List(other_best_paths), position, beta=1.0,
                                                             alpha=1.0, number_of_ants=self.NUMBER_OF_ANTS)
            else:
                cost, aco_path, updated_pheromone = optimize(map_data, pheromones, heuristics, ant_limit,
                                                             None, position, beta=1.0, alpha=1.0,
                                                             number_of_ants=self.NUMBER_OF_ANTS)

            if cost < best_cost:  # If the new other_path is better
                best_cost = cost
                best_path = aco_path

            # Update pheromone
            pheromones = updated_pheromone

        # Check for previous other_path
        if len(best_path) <= 1:
            return self.repair(path, agent, t - 1, real_t, scenario, other_paths)

        self.pheromones[agent.id] = pheromones

        return path[:t] + best_path
