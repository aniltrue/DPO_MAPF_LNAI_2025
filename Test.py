import numpy as np
from scipy.optimize import linear_sum_assignment


# Function to compute the Manhattan distance between two points
def manhattan_distance(point1, point2):
    return abs(point1[0] - point2[0]) + abs(point1[1] - point2[1])


# Function to compute the EMD between two paths
def compute_emd(path1, path2):
    n = len(path1)
    m = len(path2)

    # Create the cost matrix (distance matrix)
    cost_matrix = np.zeros((n, m))

    # Fill in the cost matrix with the Manhattan distances between each pair of points
    for i in range(n):
        for j in range(m):
            cost_matrix[i, j] = manhattan_distance(path1[i], path2[j])

    # Solve the linear sum assignment problem (Hungarian algorithm)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Calculate the total cost (sum of the optimal assignment)
    total_cost = cost_matrix[row_ind, col_ind].sum()

    # Create the flow matrix (1 for assigned pairs, 0 otherwise)
    flow_matrix = np.zeros((n, m))
    for i, j in zip(row_ind, col_ind):
        flow_matrix[i, j] = 1

    return total_cost, flow_matrix


# Example usage
path1 = [(1, 1), (2, 1), (3, 1)]  # Original path
path2 = [(1, 2), (2, 2), (3, 2)]  # Optimized path

# Compute the EMD
total_cost, flow_matrix = compute_emd(path1, path2)

print("Total EMD cost:", total_cost)
print("Flow matrix:")
print(flow_matrix)
