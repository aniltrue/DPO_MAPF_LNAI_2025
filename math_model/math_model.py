from typing import Dict, Union, Optional
import numpy as np
from gurobipy import GRB, tupledict
import gurobipy as gp
from agent.AbstractAgent import AgentID
from agent.Path import Path


def convert_to_paths(X: tupledict, number_of_agents: int, max_t: int, n: int) -> Dict[AgentID, Path]:
    """
        This method converts GRB variables to Path instance for each agent.

        :param X: GRB variables provided by Gurobi
        :param number_of_agents: Number of agents.
        :param max_t: Maximum time-step for the environment
        :param n: Map size (n x n)
        :return: Path for each agent provided by Gurobi
    """
    paths = {
        m: [] for m in range(1, number_of_agents + 1)
    }

    for t in range(max_t):
        for m in range(1, number_of_agents + 1):
            check = False

            # Find location of agent m for time-step t

            for i in range(n):
                if check:
                    break

                for j in range(n):
                    if check:
                        break

                    if round(X[i, j, t, m].X) == 1.:
                        paths[m].append((i, j))
                        check = True

    return paths


def solve(parameters: Dict[str, Union[int | np.ndarray]], t_max: Dict[AgentID, int], presolve: bool = True,
          log_out: bool = False) -> (gp.Model, Optional[Dict[AgentID, Path]], bool):
    """
        This method employs Gurobi to solve DPO-MAPF problem as an LP Problem

        :param parameters: LP parameters of the scenario
        :param t_max: Maximum time-step limitation for each agent
        :param presolve: Whether the model is *pre-solved* or *solved*. **Note**: *pre-solve* does not provide paths.
        :param log_out: Whether log to console
        :return: GRB Model, paths for each agent and whether the solution is feasible, or not.
    """

    model = gp.Model("MAPF")

    try:
        # Sets
        n = parameters["n"]
        max_t = parameters["T"]
        m = parameters["m"]

        M = range(1, m + 1)
        I = range(0, n)
        J = range(0, n)
        T = range(0, max_t)

        # Decision variables
        X = model.addVars(((i, j, t, m) for i in I for j in J for t in T for m in M), lb=0., ub=1., vtype=GRB.BINARY)
        A = model.addVars(((t, m) for t in T for m in M), lb=0., ub=1., vtype=GRB.BINARY)

        # Constraint1
        model.addConstrs(gp.quicksum(X[i, j, t, m] for m in M) <= 1 for t in T for i in I for j in J)

        # Constraint2
        model.addConstrs(X[i, j, t, m] <= 1 - parameters["o"][i, j, t] for t in T for m in M for i in I for j in J)
        model.addConstrs(
            X[i, j, t, m] <= 1 - parameters["o"][i, j, t - 1] for t in range(1, max_t) for m in M for i in I for j in J)

        # Constraint3
        model.addConstrs(gp.quicksum(X[i, j, t, m] for i in I for j in J) == 1 - A[t, m] for m in M for t in T)

        # Constraint4
        model.addConstrs(X[i, j, 0, m] == parameters["s"][i, j, m] for m in M for i in I for j in J)

        # Constraint5
        model.addConstrs(X[i, j, t, m] <=
                         gp.quicksum(X[k, j, t - 1, m] for k in range(max(0, i - 1), min(n, i + 2))) +
                         gp.quicksum(X[i, k, t - 1, m] for k in range(max(0, j - 1), min(n, j + 2)))
                         for i in I for j in J for t in range(1, max_t) for m in M)

        # Constraint6
        model.addConstrs(
            gp.quicksum(X[i, j, t, m] for t in T) >= parameters["g"][i, j, m] for i in I for j in J for m in M)

        # Constraint7
        for i in I:
            for j in J:
                for k in range(i - 1, i + 2):
                    for l in range(j - 1, j + 2):
                        if k < 0 or l < 0 or k >= n or l >= n:
                            continue

                        if abs(i - k) + abs(j - l) > 1:
                            continue

                        if i == k and j == l:
                            continue

                        model.addConstrs(X[i, j, t - 1, m] + X[k, l, t, m] + X[k, l, t - 1, n] + X[i, j, t, n] <= 3
                                         for t in range(1, max_t) for n in M for m in M if n != m)

        # Constraint8
        model.addConstrs(A[t + 1, m] <= gp.quicksum(X[i, j, t, m] * parameters["g"][i, j, m] for i in I for j in J) + A[t, m] for t in range(0, max_t - 1) for m in M)

        # Constraint 9 and Objective
        travel_time = {m: model.addVar(lb=0., ub=t_max[m], obj=0, column=None, vtype=GRB.CONTINUOUS,
                                       name=f"TravelTime_{m}") for m in M}

        model.addConstrs(travel_time[m] == gp.quicksum(X[i, j, t, m]
                                                       for i in I for j in J for t in range(1, max_t)) for m in M)

        model.setObjective(gp.quicksum(travel_time[m] for m in M), GRB.MINIMIZE)

        # Parameters
        model.setParam('TimeLimit', 60 * 60)
        model.setParam('Heuristics', 0.1)
        model.setParam('NoRelHeurTime', 20)

        if not log_out:
            model.setParam('LogToConsole', 0)
        else:
            model.setParam('LogToConsole', 1)

        model.update()

        if presolve:
            model = model.presolve()

            model.optimize()

            return model, None, True

        model.optimize()

        return model, convert_to_paths(X, m, max_t, n), True

    except Exception as e:  # Model is infeasible
        if log_out:
            print(str(e))
    
        return model, None, False
