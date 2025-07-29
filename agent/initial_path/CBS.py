import os.path
import subprocess
from typing import List, Dict

import numpy as np

from env.map import AbstractMap
from .InitialPathStrategy import AbstractInitialPathStrategy, AgentID, Path
from agent.utils import find_current_and_goal_points


class CBS(AbstractInitialPathStrategy):
    MAP_NAME = "map.map"
    SCENARIO_NAME = "scenario.scen"

    @property
    def name(self) -> str:
        return "CBSH2-RTC"

    def to_benchmark(self, scenario: AbstractMap, agent_ids: List[AgentID]):
        map_data = scenario.get_raw_data()

        width, height = map_data.shape[0], map_data.shape[1]

        with open(os.path.join("agent/initial_path/", self.MAP_NAME), "w") as f:
            f.write("type octile\n")
            f.write(f"height {height}\n")
            f.write(f"width {width}\n")
            f.write("map\n")

            for i in range(height):
                for j in range(width):
                    f.write("." if map_data[i, j, 0] == 0 else "@")

                f.write("\n")

        with open(os.path.join("agent/initial_path/", self.SCENARIO_NAME), "w") as f:
            f.write("version 1\n")
            for agent_id in agent_ids:
                (start_i, start_j), (goal_i, goal_j) = find_current_and_goal_points(map_data, agent_id)

                optimal_path = np.sqrt(np.power(start_i - goal_i, 2.) + np.power(start_j - goal_j, 2.))

                f.write("\t".join([str(0), self.MAP_NAME, str(height), str(width), str(start_j), str(start_i),
                                   str(goal_j), str(goal_i), str(optimal_path)]) + "\n")

    @staticmethod
    def read_paths(agent_ids: List[AgentID]) -> Dict[AgentID, Path]:
        paths = {agent_id: [] for agent_id in agent_ids}

        with open(os.path.join("agent/initial_path/", "paths.txt"), "r") as f:
            lines = f.readlines()

            for i in range(len(agent_ids)):
                values = lines[i].split(":")[-1].split("->")[:-1]

                for value in values:
                    coordinates = value.strip().replace("(", "").replace(")", "").split(",")
                    paths[agent_ids[i]].append((int(coordinates[0]), int(coordinates[1])))

        return paths

    def generate_initial_paths(self, scenario: AbstractMap, agent_ids: List[AgentID], t_max: Dict[AgentID, int],
                               **kwargs) -> Dict[AgentID, Path]:
        self.to_benchmark(scenario, agent_ids)

        command = [
            "agent\\initial_path\\cbs.exe",
            "-m", "agent\\initial_path\\" + self.MAP_NAME,
            "-a", "agent\\initial_path\\" + self.SCENARIO_NAME,
            "-o", "agent\\initial_path\\output.csv",
            "--outputPaths=agent\\initial_path\\paths.txt",
            "-k", str(len(agent_ids)),
            "-t", str(min(t_max.values()))
        ]

        paths = {agent_id: [] for agent_id in agent_ids}
        try:
            run_result = subprocess.run(command, capture_output=True, text=True, check=True)
            # print(f"EECBS output: {run_result.stdout}")

            if not os.path.exists(os.path.join("agent/initial_path/", "paths.txt")) or \
                    not os.path.exists(os.path.join("agent/initial_path/", "output.csv")):
                raise Exception("No solution found! (CBS)")

            paths = self.read_paths(agent_ids)

        except Exception as e:
            print(str(e))
            # print(f"Return code: {e.returncode}")
            # print(f"Command: {e.cmd}")

            for agent_id in agent_ids:
                (start_i, start_j), _ = find_current_and_goal_points(scenario.get_raw_data(), agent_id)

                paths[agent_id].append((start_i, start_j))

        if os.path.exists(os.path.join("agent/initial_path/", self.SCENARIO_NAME)):
            os.remove(os.path.join("agent/initial_path/", self.SCENARIO_NAME))

        if os.path.exists(os.path.join("agent/initial_path/", self.MAP_NAME)):
            os.remove(os.path.join("agent/initial_path/", self.MAP_NAME))

        if os.path.exists(os.path.join("agent/initial_path/", "paths.txt")):
            os.remove(os.path.join("agent/initial_path/", "paths.txt"))

        if os.path.exists(os.path.join("agent/initial_path/", "output.csv")):
            os.remove(os.path.join("agent/initial_path/", "output.csv"))

        return paths
