import datetime
import glob
import random

import numpy as np

import env
from agent.AgentFactory import AgentFactory, AbstractMap
from agent.conflict_resolution import *
from agent.initial_path import EECBS, CBS, AbstractInitialPathStrategy
from agent.initial_path.aco import set_numba_seed
from agent.repairing import *
from env.Environment import Environment
import pandas as pd


INITIAL_ALGS = [EECBS, CBS]
REPAIR_ALGS = [ACOUninformedRepairingStrategy, ACOInformedRepairingStrategy, ACOWAPRepairingStrategy, ACOInformedWAPRepairingStrategy, EmptyRepairing]
PROTOCOLS = [FairTokenProtocol, ProbabilityBasedProtocol]
AGENT_SEEDS = [1, 2, 3, 4, 5]
MAP_SEEDS = [1, 2, 3]

df = pd.DataFrame(
    columns=["MapSize", "NumberOfAgents", "Density", "MapSeed", "AgentSeed", "IsDynamic", "InitialAlg", "RepairingAlg",
             "Protocol", "Feasibility", "FirstTierElapsedTime", "Result", "TimeStep", "TotalElapsedTime", "Cost",
             "OptimalValue", "OptimalElapsedTime", "NormalizedCost", "MaxDiffTokens", "LowerBound", "AverageEMD",
             "SecondTierElapsedTime", "MapName", "ScenName", "SubsetID"])

FILE_PATH = "evaluation_benchmark.xlsx"

counter = 0


def run(map_obj: AbstractMap, initial_alg: AbstractInitialPathStrategy, repairing_alg: AbstractRepairingStrategy, protocol_alg: AbstractPriorityProtocol, row_exp: dict) -> dict:
    agents, elapsed_time = AgentFactory.generate(map_obj.clone(), initial_alg,
                                                 repairing_alg, protocol_alg,
                                                 max_iteration=150,
                                                 initial_b=5.0,
                                                 end_b=0.5,
                                                 number_of_ants=75)

    Env = Environment(map_obj.clone(), agents)

    result = Env.run()

    row = row_exp.copy()

    row["InitialAlg"] = initial_alg.name
    row["RepairingAlg"] = repairing_alg.name
    row["Protocol"] = protocol_alg.name

    row["Feasibility"] = result["feasibility"]
    row["FirstTierElapsedTime"] = elapsed_time
    row["Result"] = result["result"]
    row["TimeStep"] = result["time_step"]
    row["TotalElapsedTime"] = elapsed_time + result["elapsed_time"]
    row["Cost"] = result["cost"]
    row["NormalizedCost"] = result["normalized_cost"]
    row["MaxDiffTokens"] = result["max_token_diff"]
    row["LowerBound"] = result["lower_bound"]
    row["SecondTierElapsedTime"] = result["elapsed_time"]
    row["AverageEMD"] = result["average_emd"]

    return row


for map_file in glob.glob("benchmark_data/map/*.map"):
    map_file = map_file.replace("\\", "/")

    if map_file.split('/')[-1][:-4].startswith("maze"):
        continue

    scen_query = "benchmark_data/scenario/" + map_file.split('/')[-1][:-4] + "-even-*.scen"

    try:
        wrapper = env.map.BenchmarkMapWrapper(map_file)
    except Exception as e:
        print(e, map_file)

        continue

    for scen_file in glob.glob(scen_query):
        for map_seed in MAP_SEEDS:
            org_map, number_of_agent, target_subset = wrapper(scen_file, map_seed)

            row_exp = {"MapSize": org_map.n,
                       "NumberOfAgents": number_of_agent,
                       "Density": "-",
                       "MapSeed": map_seed,
                       "IsDynamic": True,
                       "OptimalValue": "-",
                       "OptimalElapsedTime": "-",
                       "MaxDiffTokens": 0,
                       "AverageEMD": 0,
                       "MapName": map_file.split('/')[-1][:-4],
                       "ScenName": scen_file.split('/')[-1][:-5],
                       "SubsetID": target_subset}

            for agent_seed in AGENT_SEEDS:
                row_exp["AgentSeed"] = agent_seed
                random.seed(agent_seed)
                np.random.seed(agent_seed)
                set_numba_seed(agent_seed)

                for initial_alg_class in INITIAL_ALGS:
                    for repairing_alg_class in REPAIR_ALGS:
                        for protocol_alg_class in PROTOCOLS:
                            initial_alg = initial_alg_class()
                            repairing_alg = repairing_alg_class()
                            protocol_alg = protocol_alg_class()

                            row = run(org_map, initial_alg, repairing_alg, protocol_alg, row_exp)

                            df.loc[counter] = row
                            counter += 1

            print(map_file, map_seed, scen_file, datetime.datetime.now())

            df.to_excel(FILE_PATH, sheet_name="EvaluationData")
