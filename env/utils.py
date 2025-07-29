from typing import List
import numpy as np
from numba import njit


@njit
def check_for_collision(map_data: np.ndarray, obstacle_axis: List[int], mask_axis: int = -1) -> bool:
    """
        This method leveraging *Numba* checks for any collision.

        :param map_data: The scenario data as *NumPy* array
        :param obstacle_axis: The list of axis where obstacles are defined
        :param mask_axis: Partial observability mask axis. Default value is -1 which means fully observable.
        :return: Whether any collision is detected, or not
    """

    n: int = map_data.shape[0]

    for i in range(n):
        for j in range(n):
            if map_data[i, j, 1] == 0:  # Is agent there?
                continue

            if mask_axis != -1 and map_data[i, j, mask_axis] == 1:  # Is the cell visible?
                continue

            for axis in obstacle_axis:  # Check for obstacles
                if map_data[i, j, axis] != 0:  # Is obstacle there?
                    return True

    return False


@njit
def move_dynamic_obstacles(map_data: np.ndarray, dynamic_obstacle_axis: int = 3,
                           up: int = 1,
                           down: int = 2,
                           right: int = 3,
                           left: int = 4):
    """
        This method leveraging *Numba* moves dynamic obstacles.

        :param map_data: The scenario data as *NumPy* array
        :param dynamic_obstacle_axis: The axis where the dynamic obstacles are presented.
        :param up: **Up** move value, it must be non-zero.
        :param down: **Down** move value, it must be non-zero.
        :param right: **Right** move value, it must be non-zero.
        :param left: **Left** move value, it must be non-zero.
    """

    n: int = map_data.shape[0]

    next_dynamic_obstacle_map = np.zeros((n, n), dtype=np.int32)

    for i in range(n):
        for j in range(n):
            if map_data[i, j, dynamic_obstacle_axis] == 0:  # No obstacle
                continue

            move = map_data[i, j, dynamic_obstacle_axis]
            next_move = move
            next_i = i
            next_j = j

            if move == up:
                if i == 0 or map_data[i - 1, j, 0] != 0:
                    next_move = down
                else:
                    next_i -= 1
            elif move == down:
                if i == n - 1 or map_data[i + 1, j, 0] != 0:
                    next_move = up
                else:
                    next_i += 1
            elif move == right:
                if j == n - 1 or map_data[i, j + 1, 0] != 0:
                    next_move = left
                else:
                    next_j += 1
            elif move == left:
                if j == 0 or map_data[i, j - 1, 0] != 0:
                    next_move = right
                else:
                    next_j -= 1

            next_dynamic_obstacle_map[next_i, next_j] = next_move

    # Update current scenario
    map_data[:, :, dynamic_obstacle_axis] = next_dynamic_obstacle_map


@njit
def update_vision(map_data: np.ndarray, mask_axis: int = 4, vision_range: int = 2):
    """
        This method leveraging *Numba* updates the vision for the partial observability.

        :param map_data: The scenario data as *NumPy* array
        :param mask_axis: The axis where the vision mask is presented.
        :param vision_range: The range of vision for the agents.
    """

    n: int = map_data.shape[0]

    map_data[:, :, mask_axis] = 1  # Remove old vision

    for i in range(n):
        for j in range(n):
            if map_data[i, j, 1] == 0:  # Is agent there?
                continue

            # Set vision/Remove mask
            for v_i in range(max(i - vision_range, 0), min(i + vision_range + 1, n)):
                for v_j in range(max(j - vision_range, 0), min(j + vision_range + 1, n)):
                    map_data[v_i, v_j, mask_axis] = 0
