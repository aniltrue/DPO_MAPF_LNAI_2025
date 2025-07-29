import math

import numba
from typing import List, Dict
import numpy as np
from env.map import AbstractMap
from .InitialPathStrategy import AbstractInitialPathStrategy
from .aco import optimize, get_heuristic_matrix
from agent.AbstractAgent import AgentID
from agent.Path import Path
from threading import Thread


class MAACO2(AbstractInitialPathStrategy):
    """
            This Obstacle-Based MA-ACO sets a static obstacle where the other agents pass
    """

    INITIAL_BETA: float = 2.0  #: Initial beta value for dynamic beta adjustment
    FINAL_BETA: float = 0.1  #: Final beta value for dynamic beta adjustment

    pheromones: Dict[AgentID, np.ndarray]  #: Pheromone Matrix
    heuristics: Dict[AgentID, np.ndarray]  #: Heuristic Information Matrix

    @property
    def name(self) -> str:
        return "Obstacle-Based MA-ACO"

    def generate_initial_paths(self, scenario: AbstractMap, agent_ids: List[AgentID], t_max: Dict[AgentID, int], **kwargs) \
            -> Dict[AgentID, Path]:

        # Initiate variables
        best_paths = {agent_id: [] for agent_id in agent_ids}
        best_costs = {agent_id: t_max[agent_id] + 1 for agent_id in agent_ids}
        pheromones = {agent_id: np.ones((scenario.n, scenario.n)) for agent_id in agent_ids}
        heuristics = {agent_id: get_heuristic_matrix(scenario.get_data(), agent_id, t_max[agent_id])
                      for agent_id in agent_ids}

        # Get max. iteration
        assert "max_iteration" in kwargs, "Max. Iteration must be defined."

        max_iteration: int = int(kwargs["max_iteration"])
        del kwargs["max_iteration"]

        # Get Beta parameters
        assert "initial_b" in kwargs, "Initial beta must be defined."

        initial_beta: float = float(kwargs["initial_b"])

        assert "end_b" in kwargs, "End beta must be defined."

        end_beta: float = float(kwargs["end_b"])

        # Get number of ants
        assert "number_of_ants" in kwargs, "Number of Ants must be defined."

        number_of_ants: int = int(kwargs["number_of_ants"])

        # Initiate beta parameter for dynamic adjustment
        beta = initial_beta
        beta_decay = (beta - end_beta) / max_iteration

        # Initiate scenario data
        map_data = scenario.get_raw_data()

        # Get positions
        self.find_positions(map_data, agent_ids)

        # Early stopping threshold
        non_improve_threshold = math.ceil(max_iteration / 3)
        non_improve_counters = {agent_id: 0 for agent_id in agent_ids}

        for i in range(max_iteration):  # Start iteration
            # Prepare `other best paths` for each agent
            other_paths = {agent_id: [] for agent_id in agent_ids}
            for agent_id in agent_ids:
                if non_improve_counters[agent_id] > non_improve_threshold:
                    continue
                
                # Collect other best paths
                other_best_paths: List[Path] = []

                for other_agent_id in agent_ids:
                    if other_agent_id != agent_id and len(best_paths[other_agent_id]) > 0:
                        other_best_paths.append(best_paths[other_agent_id])

                other_paths[agent_id] = other_best_paths

            threads = []
            for agent_id in agent_ids:
                if non_improve_counters[agent_id] > non_improve_threshold:
                    continue
                
                def single_aco(id: int):  # Single-ACO as a function
                    # Run ACO
                    if len(other_paths[id]) > 0:
                        cost, path, updated_pheromone = optimize(map_data, pheromones[id], heuristics[id], t_max[id],
                                                                 numba.typed.List(other_paths[id]),
                                                                 self.positions[agent_id], beta=beta, alpha=1.0,
                                                                 number_of_ants=number_of_ants)
                    else:
                        cost, path, updated_pheromone = optimize(map_data, pheromones[id], heuristics[id], t_max[id],
                                                                 None, self.positions[agent_id], beta=beta, alpha=1.0,
                                                                 number_of_ants=number_of_ants)

                    if cost < best_costs[id]:  # If the new other_path is better
                        best_costs[id] = cost
                        best_paths[id] = path
                        non_improve_counters[agent_id] = 0
                    else:
                        non_improve_counters[agent_id] += 1

                    # Update pheromone
                    pheromones[id] = updated_pheromone

                # Multi-Thread Single ACO
                thread = Thread(target=single_aco, args=(agent_id,))
                thread.start()
                threads.append(thread)

            # Join threads
            for thread in threads:
                thread.join()

            # Decrease beta value
            beta -= beta_decay

        # Save the heuristic information and pheromones
        self.pheromones = pheromones
        self.heuristics = heuristics

        return best_paths
