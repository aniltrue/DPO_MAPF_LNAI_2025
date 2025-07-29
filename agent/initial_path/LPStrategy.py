from typing import List, Dict

from .InitialPathStrategy import AbstractInitialPathStrategy
from agent import AgentID, Path
from env.map import AbstractMap
from math_model import LPFactory


class LPStrategy(AbstractInitialPathStrategy):
    @property
    def name(self) -> str:
        return "LP Strategy"

    def generate_initial_paths(self, scenario: AbstractMap, agent_ids: List[AgentID], t_max: Dict[AgentID, int],
                               **kwargs) -> Dict[AgentID, Path]:
        return LPFactory.solve(scenario)
