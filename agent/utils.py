from typing import Tuple, List
import numpy as np
from numba import njit
from .AbstractAgent import AgentID
from .Path import Coordinate


@njit
def find_current_and_goal_points(map_data: np.ndarray, agent_id: AgentID) -> Tuple[Coordinate, Coordinate]:
    """
        This method finds the current and goal points of the given agent.

        :param map_data: Current scenario data
        :param agent_id: Target agent's id
        :return: Current and Goal coordinates
    """
    current_i = -1
    current_j = -1

    goal_i = -1
    goal_j = -1

    n = map_data.shape[0]

    for i in range(n):
        for j in range(n):
            if map_data[i, j, 1] == agent_id:   # Check for current point
                current_i = i
                current_j = j

            if map_data[i, j, 2] == agent_id:   # Check for goal point
                goal_i = i
                goal_j = j

            if goal_i >= 0 and current_i >= 0:  # If both current and goal points are found
                return (current_i, current_j), (goal_i, goal_j)

    return (current_i, current_j), (goal_i, goal_j)


@njit
def quicksort(arr: list) -> List[int]:
    """
        This method employing *QuickSort* algorithm provides the sorted **indices** of a given list

        :param arr: Target list
        :return: Sorted indices
    """
    indices = list(range(len(arr)))

    stack = [(0, len(indices) - 1)]

    while stack:
        low, high = stack.pop(-1)

        if low < high:
            pivot_index = partition(arr, indices, low, high)

            # Push the smaller sublist to the stack first
            if pivot_index - low < high - pivot_index:
                stack.append((pivot_index + 1, high))
                stack.append((low, pivot_index - 1))
            else:
                stack.append((low, pivot_index - 1))
                stack.append((pivot_index + 1, high))

    return indices


@njit
def partition(arr: list, indices: List[int], low: int, high: int) -> int:
    """
        Partition approach for *Quick Sort*

        :param arr: The given list to be sorted
        :param indices: Current list of indices
        :param low: Low index of the sublist
        :param high: High index of the sublist
        :return: New pivot index
    """
    if high - low <= 0:
        return low

    medium = (high + low) // 2
    pivot_index = high

    # Select a better pivot to split
    if arr[indices[low]] <= arr[indices[medium]] <= arr[indices[high]]:
        pivot_index = medium
    elif arr[indices[medium]] <= arr[indices[low]] <= arr[indices[high]]:
        pivot_index = low

    # Swap the pivot
    indices[high], indices[pivot_index] = indices[pivot_index], indices[high]

    # Start partition
    pivot = arr[indices[high]]
    i = low
    j = high - 1

    while i < j:
        while arr[indices[i]] < pivot and i < j:
            i += 1
        while arr[indices[j]] >= pivot and i < j:
            j -= 1

        indices[i], indices[j] = indices[j], indices[i]

        if i < j:
            i += 1
            j -= 1

    # Swap back
    indices[i], indices[high] = indices[high], indices[i]

    return i
