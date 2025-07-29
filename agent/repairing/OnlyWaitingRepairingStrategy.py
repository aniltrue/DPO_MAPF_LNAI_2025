import math
from typing import Optional, List, Dict
import numba
import numpy as np
from env.map import AbstractMap
from .RepairingStrategy import AbstractRepairingStrategy
from agent.Path import Path, Coordinate
from agent.AbstractAgent import AgentID, AbstractAgent
from agent.initial_path.aco import optimize, get_heuristic_matrix


class OnlyWaitingRepairingStrategy(AbstractRepairingStrategy):
    """
        This repairing approach employs only *WAITING* strategy
    """

    def __init__(self):
        super().__init__()
    @property
    def name(self) -> str:
        return "Only Waiting Repairing Strategy"

    def repair(self, path: Path, agent: AbstractAgent, t: int, real_t: int, scenario: AbstractMap,
               other_paths: Optional[List[Path]]) -> Optional[Path]:

        if t < real_t:
            return None

        # Waiting strategy
        path_with_waiting = self.waiting_strategy(agent, t, real_t, scenario)

        if path_with_waiting is not None:
            if 1 < t < len(path):
                return path_with_waiting

        # Check for previous other_path
        return self.repair(path, agent, t - 1, real_t, scenario, other_paths)
