import datetime
import time

import numpy as np

import env
from agent.AgentFactory import AgentFactory
from agent.conflict_resolution import RandomProtocol, FairTokenProtocol, ProbabilityBasedProtocol
from agent.initial_path import EECBS, CBS
from agent.initial_path.aco import set_numba_seed
from agent.repairing import *
from env.Environment import Environment
import random
import pandas as pd

MAP_SIZES = [30, 50, 100, 200]
AGENTS = [10, 15, 20, 25]
DENSITIES = [0.05, 0.1, 0.15, 0.2]
MAP_SEEDS = [1, 2, 3]
AGENT_SEEDS = [1, 2, 3, 4, 5]

INITIAL_ALGS = [CBS, EECBS]
REPAIR_ALGS = [EmptyRepairing, ACOUninformedRepairingStrategy, ACOInformedRepairingStrategy, ACOWAPRepairingStrategy, ACOInformedWAPRepairingStrategy]
PROTOCOLS = [RandomProtocol, FairTokenProtocol, ProbabilityBasedProtocol]

FILE_PATH = "large_instances.xlsx"


df = pd.DataFrame(columns=["MapSize", "NumberOfAgents", "Density", "MapSeed", "AgentSeed", "IsDynamic", "InitialAlg", "RepairingAlg", "Protocol", "Feasibility", "Result", "TimeStep", "ElapsedTime", "Cost", "NormalizedCost", "MaxDiffTokens", "LowerBound", "AverageEMD"])

# df = pd.read_excel(FILE_PATH, sheet_name="EvaluationData")

counter = len(df) + 1

for map_size in MAP_SIZES:
    for number_of_agent in AGENTS:
        for density in DENSITIES:
            if map_size == 30 and number_of_agent == 25:
                continue

            for map_seed in MAP_SEEDS:
                map_seed_counter = 0
                org_map = env.map.StaticMap.random_map_factory(map_size, number_of_agent, density, map_seed + map_seed_counter * 100)

                for agent_seed in AGENT_SEEDS:
                    row_exp = {"MapSize": map_size,
                               "NumberOfAgents": number_of_agent,
                               "Density": density,
                               "AgentSeed": agent_seed,
                               "IsDynamic": False,
                               "MaxDiffTokens": 0,
                               "AverageEMD": 0}

                    for initial_alg in INITIAL_ALGS:
                        for repair_alg in REPAIR_ALGS:
                            for protocol_type in PROTOCOLS:
                                random.seed(agent_seed)
                                np.random.seed(agent_seed)
                                set_numba_seed(agent_seed)

                                initial_algorithm = initial_alg()
                                repairing_algorithm = repair_alg()
                                protocol = protocol_type()

                                if initial_algorithm.name in ["CBSH2-RTC", "EECBS"]:
                                    if repairing_algorithm.name != "No-Repairing" or protocol.name != "Random Protocol":
                                        continue
                                    if agent_seed > 1:
                                        continue

                                elif repairing_algorithm.name == "No-Repairing":
                                    continue

                                if initial_algorithm.name == "CBSH2-RTC":
                                    while True:
                                        org_map = env.map.StaticMap.random_map_factory(map_size, number_of_agent, density, map_seed + map_seed_counter * 100)
                                        agents, elapsed_time = AgentFactory.generate(org_map, initial_algorithm,
                                                                                     repairing_algorithm, protocol,
                                                                                     max_iteration=150,
                                                                                     initial_b=5.0,
                                                                                     end_b=0.5,
                                                                                     number_of_ants=75)

                                        Env = Environment(org_map.clone(), agents)

                                        result = Env.run()

                                        if result["feasibility"]:
                                            print("Started: ", map_size, number_of_agent, density,
                                                  map_seed + map_seed_counter * 100, datetime.datetime.now())
                                            break
                                        else:
                                            map_seed_counter += 1
                                else:
                                    org_map = env.map.StaticMap.random_map_factory(map_size, number_of_agent, density,
                                                                                   map_seed + map_seed_counter * 100)

                                    agents, elapsed_time = AgentFactory.generate(org_map, initial_algorithm,
                                                                                 repairing_algorithm, protocol,
                                                                                 max_iteration=150,
                                                                                 initial_b=5.0,
                                                                                 end_b=0.5,
                                                                                 number_of_ants=75)

                                    Env = Environment(org_map.clone(), agents)

                                    result = Env.run()

                                row = row_exp.copy()
                                row["InitialAlg"] = initial_algorithm.name
                                row["RepairingAlg"] = repairing_algorithm.name
                                row["Protocol"] = protocol.name

                                row["MapSeed"] = map_seed + map_seed_counter * 100

                                row["Feasibility"] = result["feasibility"]
                                row["Result"] = result["result"]
                                row["TimeStep"] = result["time_step"]
                                row["ElapsedTime"] = result["elapsed_time"] + elapsed_time
                                row["Cost"] = result["cost"]
                                row["NormalizedCost"] = result["normalized_cost"]
                                row["MaxDiffTokens"] = result["max_token_diff"]
                                row["LowerBound"] = result["lower_bound"]
                                row["AverageEMD"] = result["average_emd"]

                                df.loc[counter] = row
                                counter += 1

                print("DPO Started.")

                df.to_excel(FILE_PATH, sheet_name="EvaluationData")

                org_map = env.map.DPOMap.random_map_factory(map_size, number_of_agent, density,
                                                            map_seed + map_seed_counter * 100)

                for agent_seed in AGENT_SEEDS:
                    row_exp = {"MapSize": map_size,
                               "NumberOfAgents": number_of_agent,
                               "Density": density,
                               "AgentSeed": agent_seed,
                               "IsDynamic": True,
                               "MaxDiffTokens": 0,
                               "AverageEMD": 0}

                    for initial_alg in INITIAL_ALGS:
                        for repair_alg in REPAIR_ALGS:
                            for protocol_type in PROTOCOLS:
                                random.seed(agent_seed)
                                np.random.seed(agent_seed)
                                set_numba_seed(agent_seed)

                                initial_algorithm = initial_alg()
                                repairing_algorithm = repair_alg()
                                protocol = protocol_type()

                                if repairing_algorithm.name == 'No-Repairing' and protocol.name != 'Random Protocol':
                                    continue
                                if repairing_algorithm.name != 'No-Repairing' and protocol.name == 'Random Protocol':
                                    continue

                                agents, elapsed_time = AgentFactory.generate(org_map.clone(), initial_algorithm,
                                                                             repairing_algorithm, protocol,
                                                                             max_iteration=150,
                                                                             initial_b=5.0,
                                                                             end_b=0.25,
                                                                             number_of_ants=50)

                                Env = Environment(org_map.clone(), agents)

                                result = Env.run()

                                row = row_exp.copy()
                                row["InitialAlg"] = initial_algorithm.name
                                row["RepairingAlg"] = repairing_algorithm.name
                                row["Protocol"] = protocol.name

                                row["MapSeed"] = map_seed + map_seed_counter * 100

                                row["Feasibility"] = result["feasibility"]
                                row["Result"] = result["result"]
                                row["TimeStep"] = result["time_step"]
                                row["ElapsedTime"] = result["elapsed_time"] + elapsed_time
                                row["Cost"] = result["cost"]
                                row["NormalizedCost"] = result["normalized_cost"]
                                row["MaxDiffTokens"] = result["max_token_diff"]
                                row["LowerBound"] = result["lower_bound"]
                                row["AverageEMD"] = result["average_emd"]

                                df.loc[counter] = row
                                counter += 1

                print(map_size, number_of_agent, density, map_seed, datetime.datetime.now())

                df.to_excel(FILE_PATH, sheet_name="EvaluationData")

df.to_excel(FILE_PATH, sheet_name="EvaluationData")
