from typing import Optional, List
from env.map import AbstractMap
from .Path import Path, Coordinate
from .Agent import AbstractAgent
from .conflict_resolution import AbstractPriorityProtocol
from .repairing import AbstractRepairingStrategy


class LPAgent(AbstractAgent):
    repairing_strategy: AbstractRepairingStrategy   #: **Repairing Strategy** to fix a other_path when a conflict occurs
    priority_protocol: AbstractPriorityProtocol     #: **Priority Protocol** determines the priority of agents in case of conflict between multiple agents.
    other_agents: List[AbstractAgent]               #: Other agents

    def __init__(self, agent_id: int, t_max: int, repairing_strategy: AbstractRepairingStrategy,
                 priority_protocol: AbstractPriorityProtocol, path: Optional[Path] = None):
        super().__init__(agent_id, t_max, path)

        self.repairing_strategy = repairing_strategy
        self.priority_protocol = priority_protocol
        self.other_agents = []

    def resolve(self, agents: Optional[List[AbstractAgent]], scenario: AbstractMap, t: int,
                real_t: Optional[int] = None):

        pass

    def update(self, scenario: AbstractMap, t: int):
        pass
