import datetime
import random
import time
import numpy as np
import env
from agent.AgentFactory import AgentFactory
from agent.conflict_resolution import RandomProtocol, FairTokenProtocol, ProbabilityBasedProtocol
from agent.initial_path import MAACO1, MAACO2, MAACO0, EECBS, CBS, LPStrategy
from agent.initial_path.aco import set_numba_seed
from agent.repairing import ACOInformedRepairingStrategy, EmptyRepairing, ACOUninformedRepairingStrategy, \
    ACOInformedWAPRepairingStrategy
from env.Environment import Environment
from math_model import LPFactory
import argparse

INITIAL_ALGS = {
    "single_aco": MAACO0,
    "pheromone_based": MAACO1,
    "obstacle_based": MAACO2,
    "EECBS": EECBS,
    "CBS": CBS,
    "lp": LPStrategy
}

PROTOCOLS = {
    "random": RandomProtocol,
    "token": FairTokenProtocol,
    "probability": ProbabilityBasedProtocol
}

REPAIRINGS = {
    "informed_wap": ACOInformedWAPRepairingStrategy,
    "informed": ACOInformedRepairingStrategy,
    "uninformed": ACOUninformedRepairingStrategy,
    "no-repairing": EmptyRepairing
}


def density_type(val) -> float:
    try:
        f_val = float(val)

        assert 0. <= f_val <= 1., f"Invalid density value ({val}). It must be in range [0.0, 1.0]"

        return f_val
    except Exception as e:
        raise argparse.ArgumentTypeError(str(e))


def positive_int(val) -> int:
    try:
        i_val = int(val)
        assert 0 < i_val, f"The value ({val}) must be a positive integer."

        return i_val
    except Exception as e:
        raise argparse.ArgumentTypeError(str(e))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Single Simulation")
    parser.add_argument("initial_algorithm", type=str, choices=list(INITIAL_ALGS.keys()),
                        help="Initial Algorithm for Centralized & Offline Path Planning")
    parser.add_argument("repairing", type=str, choices=list(REPAIRINGS.keys()),
                        help="Repairing Algorithm for Conflict Resolution")
    parser.add_argument("protocol", type=str, choices=list(PROTOCOLS.keys()),
                        help="Protocols for Conflict Resolution between Agents")
    parser.add_argument("-agent_seed", type=int, help="Random seed for ACO. If not specified, it will be random.",
                        metavar="123", default=None)

    subparsers = parser.add_subparsers(title="Map", description="Loading an existing map or generate a random map.",
                                       required=True, dest="map")

    loading_parser = subparsers.add_parser("load", help="Loading an existing map.")
    loading_parser.add_argument("other_path", type=str, help="The pickle file other_path where the map exists.",
                                metavar="map.pkl",
                                default="")

    generating_parser = subparsers.add_parser("create", help="Generate a random map.")
    generating_parser.add_argument("-n", "-map_size", type=positive_int,
                                   help="Generate a random map with given map size (n x n)", metavar="5", required=True)
    generating_parser.add_argument("-m", "-number_of_agents", type=positive_int,
                                   help="Generate a random map with given number of agents", metavar="3", required=True)
    generating_parser.add_argument("-number_of_dynamic_obstacles", type=positive_int,
                                   help="Generate a random map with given number of agents. If not specified, it will equal to number of agents (m)",
                                   metavar="3")
    generating_parser.add_argument("-d", "-density", type=density_type,
                                   help="Generate a random map with static obstacle density", metavar="0.1",
                                   default=None, required=True)
    generating_parser.add_argument("-map_seed", type=int,
                                   help="Random seed for map generation. If not specified, it will be random.",
                                   metavar="123", default=None)

    parser.add_argument("--display", action="store_false",
                        help="Whether the simulation will be displayed, or not. Default True")
    parser.add_argument("--gurobi", action="store_true",
                        help="Whether the optimal value will be found via Gurobi, or not. Default False")
    parser.add_argument("--static", action="store_false", help="Whether the map will be static, or DPO. Default False")

    parser.format_help()
    loading_parser.format_help()
    generating_parser.format_help()

    arguments = parser.parse_args()

    if arguments.map == "load":
        if arguments.static:
            scenario = env.map.DPOMap.load_map_factory(arguments.path)
        else:
            scenario = env.map.StaticMap.load_map_factory(arguments.path)
    else:
        if arguments.static:
            scenario = env.map.DPOMap.random_map_factory(arguments.n, arguments.m, arguments.d, arguments.map_seed,
                                                         number_of_dynamic_obstacles=arguments.number_of_dynamic_obstacles)
        else:
            scenario = env.map.StaticMap.random_map_factory(arguments.n, arguments.m, arguments.d, arguments.map_seed)

    if arguments.gurobi:
        start_time = time.time()
        feasibility, cost = LPFactory.solve_for_objective(scenario.clone())
        end_time = time.time()

        print("Feasibility:", feasibility,
              "Optimal Cost:", cost,
              "Elapsed Time:", str(datetime.timedelta(seconds=end_time - start_time)))

    if arguments.agent_seed is not None:
        random.seed(arguments.agent_seed)
        np.random.seed(arguments.agent_seed)
        set_numba_seed(arguments.agent_seed)

    agents, elapsed_time = AgentFactory.generate(scenario,
                                                 INITIAL_ALGS[arguments.initial_algorithm](),
                                                 REPAIRINGS[arguments.repairing](max_iter=150,
                                                                                 initial_b=5.0,
                                                                                 end_b=0.5,
                                                                                 number_of_ants=75),
                                                 PROTOCOLS[arguments.protocol](),
                                                 initial_b=5.0,
                                                 end_b=0.5,
                                                 number_of_ants=75,
                                                 max_iteration=150)

    print("Initial Path is found in", str(datetime.timedelta(seconds=elapsed_time)))

    Env = Environment(scenario, agents, will_draw=arguments.display)
    time.sleep(2)
    result = Env.run()

    result["elapsed_datetime"] = str(datetime.timedelta(seconds=result["elapsed_time"]))

    print("Results:")
    for key, value in result.items():
        print(f"\t{key}: {value}")

    if Env.will_draw:
        Env.thread.join()
