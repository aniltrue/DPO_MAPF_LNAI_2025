from abc import ABC, abstractmethod
from typing import Optional, Dict, List

from agent import AgentID
from agent.Path import Coordinate
from env.map.Map import AbstractMap
from agent.repairing.RepairingStrategy import AbstractRepairingStrategy


class AbstractPriorityProtocol(ABC):
    """
        **Priority Protocols** determine the priority of agents in case of conflict between multiple agents.
    """

    repairing_strategy: Optional[AbstractRepairingStrategy]  #: Repairing strategy to fix the other_path
    token_map: Dict[AgentID, int]  #: Token values for each agent

    def __init__(self):
        self.repairing_strategy = None
        self.token_map = {}

    @property
    @abstractmethod
    def name(self) -> str:
        """
            Name of the *Priority Protocol* for logging purposes.

            :return: The name of the strategy
        """

        ...

    @abstractmethod
    def resolve(self, agents: list, scenario: AbstractMap, t: int, real_t: Optional[int] = None):
        """
            This method resolve any possible conflicts among the given agents on the detected point.
            It employs the implemented strategy to resolve it.

            :param agents: The agents who face the conflict
            :param scenario: Current scenario
            :param t: Current time-step when the collision occurs
            :param real_t: Real current time-step. If it is *None*, it will equal to *t*
        """

        ...

    def get_max_difference(self, agent_ids: List[AgentID]) -> int:
        max_diff = 0

        for i in range(len(agent_ids)):
            for j in range(i + 1, len(agent_ids)):
                diff = self.token_map.get(agent_ids[i], 1) - self.token_map.get(agent_ids[j], 1)

                max_diff = max(max_diff, abs(diff))

        return max_diff
