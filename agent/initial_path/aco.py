import random
from typing import List, Set, Tuple, Optional
import numpy as np
from numba import njit
from agent.Path import Path, Coordinate
from agent.AbstractAgent import AgentID
from agent.utils import find_current_and_goal_points, quicksort


@njit
def set_numba_seed(seed):
    np.random.seed(seed)
    random.seed(seed)


@njit
def get_heuristic_matrix(map_data: np.ndarray, agent_id: AgentID, t_max: float, epsilon: float = 1e-8):
    """
        This method generates the *heuristic information* matrix for ACO

        **Heuristic Information**: Inverse *manhattan distance* to goal point

        :param map_data: Current scenario data as *NumPy* array
        :param agent_id: The target agent id
        :param t_max: Max. time-step limitation of the agent
        :param epsilon: Epsilon value to prevent *zero division* exception
        :return: Heuristic information matrix as *NumPy* array
    """
    _, (goal_i, goal_j) = find_current_and_goal_points(map_data, agent_id)

    n = map_data.shape[0]

    heuristic_matrix = np.zeros((n, n), dtype=np.float32)

    for i in range(n):
        for j in range(n):
            heuristic_matrix[i, j] = t_max / (abs(i - goal_i) + abs(j - goal_j) + epsilon)

    return heuristic_matrix


@njit
def optimize(map_data: np.ndarray, pheromone_matrix: np.ndarray, heuristic_matrix: np.ndarray, t_max: int,
             other_paths: Optional[List[Path]], points: Tuple[Tuple[int, int], Tuple[int, int]],
             previous_path: Optional[Path] = None, weighted_average: bool = False, number_of_ants: int = 50,
             alpha: float = 1.0, beta: float = 1.0, p: float = 0.995, elitism_ratio: float = 0.2) -> Tuple[int, Path, np.ndarray]:
    """
        This method runs the **single** agent *Ant Colony Optimization* algorithm for one-iteration.

        :param map_data: Current scenario data as *NumPy* array
        :param pheromone_matrix: Current *Pheromone level* matrix
        :param heuristic_matrix: Current *Heuristic information* matrix (e.g., inverse manhattan distance)
        :param t_max: The maximum time-step limitation for the agent
        :param other_paths: The best other_path of the other agents
        :param points: Start and Goal positions for the agent
        :param previous_path: The previous path of the agent (for informed ACO)
        :param weighted_average: Utilization of weighted-averaged pheromones
        :param number_of_ants: Number of ants for ACO
        :param alpha: Alpha controls the effect of the *pheromone level* on the move selection
        :param beta: Beta controls the effect of the *heuristic information* on the move selection
        :param p: Pheromone discount rate
        :param elitism_ratio: Elitism ratio to determine the portion of the promising ants
        :return: Tuple(Cost, Best Path, Updated Pheromone Matrix)
    """
    assert number_of_ants > 1, "Invalid number of ants"
    assert alpha >= 0, "Invalid alpha value"
    assert beta >= 0, "Invalid beta value"
    assert 0 <= p <= 1, "Invalid p value"

    (start_i, start_j), (goal_i, goal_j) = points

    if weighted_average:
        weighted_averaged_pheromones = calculate_weighted_average_pheromone(pheromone_matrix)
        weights = np.power(weighted_averaged_pheromones, alpha) * np.power(heuristic_matrix, beta)
    else:
        weights = np.power(pheromone_matrix, alpha) * np.power(heuristic_matrix, beta)

    paths: List[Path] = []
    costs: List[int] = []

    if previous_path is None:
        previous_path = [(goal_i, goal_j)]  # Add only goal

    basic_previous_path = [(goal_i, goal_j)]  # The path with only goal

    # Start ants
    for ant in range(number_of_ants):
        # Half of ants utilizes real previous path, other ones go only to goal.
        previous_path_for_ant = previous_path if ant % 2 == 0 else basic_previous_path

        path = travel_ant(map_data, (start_i, start_j), (goal_i, goal_j), previous_path_for_ant, weights, t_max, other_paths)

        paths.append(path)

        if len(path) > 0 and path[-1] == (goal_i, goal_j):  # Feasible
            costs.append(len(path) - 1)
        else:
            costs.append(t_max + 1)

    # Determine elites
    indices = quicksort(costs)

    elite_paths: List[Path] = []
    number_of_elites = round(number_of_ants * elitism_ratio)

    for i in indices[:number_of_elites]:
        if costs[i] <= t_max and paths[i] not in elite_paths:
            elite_paths.append(paths[i])

    # Update pheromones
    updated_pheromone_matrix = update_pheromone(pheromone_matrix, list(elite_paths), t_max, p)

    if len(elite_paths) == 0:
        return t_max + 1, [(start_i, start_j)], updated_pheromone_matrix

    return costs[indices[0]], paths[indices[0]], updated_pheromone_matrix


@njit
def update_pheromone(pheromone_matrix: np.ndarray, paths: List[Path], t_max: float, p: float = 0.95,
                     inverse: bool = False, epsilon: float = 1e-8) -> np.ndarray:
    """
        This method updates the pheromone matrix.

        .. math::
            T_{ij} \gets (1 - p)T_{ij} + \sum_k \Delta T_{ij}^k

            \Delta T_{ij} = t_{max} / C^k_{ij}

        :param pheromone_matrix: Target pheromone matrix
        :param paths: List of elite paths. If it is *inverse* mode, this parameter becomes '*the list of negative
        paths*'.
        :param t_max: Maximum time-step limitation
        :param p: Pheromone discount rate
        :param inverse: This mode decreases the pheromone values on the given *list of negative paths*.
        :param epsilon: Epsilon value to prevent *zero division* exception
        :return: Updated pheromone matrix
    """
    if len(paths) > 0:
        if not inverse:
            pheromone_matrix = p * pheromone_matrix

        for path in paths:
            for (i, j) in path:
                if not inverse:
                    pheromone_matrix[i, j] += t_max / (len(path) + epsilon - 1)
                else:
                    pheromone_matrix[i, j] = pheromone_matrix[i, j] * p

    return pheromone_matrix


@njit
def travel_ant(map_data: np.ndarray, start_point: Coordinate, goal_point: Coordinate, target_path: Path,
               weights: np.ndarray, t_max: int, other_paths: Optional[List[Path]] = None) -> Path:
    """
        This method generates an *ant other_path*.

        :param map_data: Map data as *NumPy* array
        :param start_point: Starting point of the agent
        :param goal_point: Goal point of the agent
        :param target_path: The previous path to reach
        :param weights: The *weights* matrix
        :param t_max: The maximum time-step for the agent
        :param other_paths: The *best* other_path of the other agents. If it is provided, this agent see their best other_path as an
        obstacle.
        :return: Ant other_path
    """
    path: Path = [start_point]

    current_i, current_j = start_point
    visited_nodes: Set[Coordinate] = {(current_i, current_j)}

    for t in range(t_max):  # Until the time-step limitation
        # Check reaching the goal or target path.
        reached = False

        for position in target_path:
            if reached:
                path.append(position)
            elif current_i == position[0] and current_j == position[1]:
                reached = True

        if reached:
            return path

        candidates = []

        # Wait
        if is_point_available((current_i, current_j), map_data, path, other_paths, t):
            candidates.append((current_i, current_j, weights[current_i, current_j]))

        # Up
        if (current_i - 1, current_j) not in visited_nodes and \
                is_point_available((current_i - 1, current_j), map_data, path, other_paths, t):

            candidates.append((current_i - 1, current_j, weights[current_i - 1, current_j]))

        # Left
        if (current_i, current_j - 1) not in visited_nodes and \
                is_point_available((current_i, current_j - 1), map_data, path, other_paths, t):

            candidates.append((current_i, current_j - 1, weights[current_i, current_j - 1]))

        # Down
        if (current_i + 1, current_j) not in visited_nodes and \
                is_point_available((current_i + 1, current_j), map_data, path, other_paths, t):

            candidates.append((current_i + 1, current_j, weights[current_i + 1, current_j]))

        # Right
        if (current_i, current_j + 1) not in visited_nodes and \
                is_point_available((current_i, current_j + 1), map_data, path, other_paths, t):

            candidates.append((current_i, current_j + 1, weights[current_i, current_j + 1]))

        if len(candidates) == 0:  # If no feasible move
            return path

        elif len(candidates) == 1:  # Only one candidate
            current_i, current_j, _ = candidates[0]

            path.append((current_i, current_j))
        else:  # Randomly select one candidate based on the weight
            candidate_index = choose_random_candidate(candidates, goal_point)

            current_i, current_j, _ = candidates[candidate_index]

            path.append((current_i, current_j))

        if (current_i, current_j) not in visited_nodes:  # Add to visited nodes
            visited_nodes.add((current_i, current_j))

    return path


@njit
def is_point_available(candidate_point: Coordinate, map_data: np.ndarray, path: Path,
                       other_paths: Optional[List[Path]] = None, t: int = 0) -> bool:
    """
        This method check whether the candidate point is available, or not

        :param candidate_point: The candidate coordinate
        :param map_data: Current scenario data
        :param path: Current other_path of the ant
        :param other_paths: The *best* other_path of other agents. The ant will avoid them.
        :param t: Current time-step
        :return: Whether the candidate point is available, or not
    """
    n = map_data.shape[0]

    # Map borders
    if 0 <= candidate_point[0] <= n - 1 and 0 <= candidate_point[1] <= n - 1:
        if map_data[candidate_point[0], candidate_point[1], 0] == 1:    # Is there any static obstacle
            return False

        if map_data[candidate_point[0], candidate_point[1], 0] >= 2:    # Is there any dynamic obstacle
            if t < 2:
                return False

    else:
        return False

    # Check for the other best paths
    if other_paths is None:
        return True

    for other_path in other_paths:
        if len(other_path) <= t:
            continue

        # Collision
        if other_path[t][0] == candidate_point[0] and other_path[t][1] == candidate_point[1]:
            return False

        # Swapping
        if t > 0 and \
            other_path[t - 1][0] == candidate_point[0] and other_path[t - 1][1] == candidate_point[1] and \
            other_path[t][0] == path[-1][0] and other_path[t][1] == path[-1][1]:

            return False

    return True


@njit
def choose_random_candidate(candidates: List[Tuple[int, int, float]], goal_point: Coordinate) -> int:
    """
        This method select a candidate from the given ones based on the *weight* value.

        **Note**: If a candidate is goal point, it will be prioritized.

        :param candidates: Candidate list
        :param goal_point: Goal coordinate
        :return: The index of the selected candidate
    """

    r = np.random.random()

    p = 0

    # Prioritize goal points
    for i in range(len(candidates)):
        candidate = candidates[i]

        if candidate[0] == goal_point[0] and candidate[1] == goal_point[1]:
            return i

    # Random selection
    weights = [candidate[-1] for candidate in candidates]
    total_weights = sum(weights)

    for i in range(len(weights)):
        weight = weights[i] / total_weights

        if p <= r < p + weight:
            return i

        p += weight

    return -1


@njit
def calculate_weighted_average_pheromone(pheromones: np.ndarray, fov_size: int = 5) -> np.ndarray:
    """
        This method calculates *weighted-averaged* pheromones.
        The pheromone level will be considered as weighted average of its around.

        :param pheromones: Pheromone levels
        :param fov_size: Size of the impact area
        :return: Weighted-averaged pheromones
    """
    weighted_averaged_pheromone = np.zeros_like(pheromones)

    weight_matrix = np.zeros((fov_size, fov_size))
    for k in range(-fov_size // 2, fov_size // 2 + 1):
        for l in range(-fov_size // 2, fov_size  // 2 + 1):
            weight_matrix[k + fov_size // 2, l + fov_size // 2] = 1. / (abs(k) + abs(l) + 1.)

    weight_matrix /= np.sum(weight_matrix)

    for i in range(pheromones.shape[0]):
        for j in range(pheromones.shape[1]):
            w = 0.

            for k in range(-fov_size // 2, fov_size // 2 + 1):
                for l in range(-fov_size // 2, fov_size // 2 + 1):
                    p = 0.

                    if i + k < 0 or i + k >= pheromones.shape[0] or j + l < 0 or j + l >= pheromones.shape[1]:
                        p = pheromones[i, j]
                    else:
                        p = pheromones[i + k, j + l]

                    w += p * weight_matrix[k + fov_size // 2][l + fov_size // 2]

            weighted_averaged_pheromone[i, j] = w

    return weighted_averaged_pheromone
