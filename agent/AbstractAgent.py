from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional, TypeVar, List
from agent.Path import Path, Coordinate
from env.map import AbstractMap

AgentID = TypeVar("AgentID", bound=int)


class AbstractAgent(ABC):
    id: int                 #: Agent ID
    t_max: int              #: Maximum time-step limitation
    path: Optional[Path]    #: Path for the agent as a list of coordinates

    def __init__(self, agent_id: int, t_max: int, path: Optional[Path] = None):
        self.id = agent_id
        self.t_max = t_max
        self.path = path

    @abstractmethod
    def resolve(self, agents: Optional[List[AbstractAgent]], scenario: AbstractMap, t: int,
                real_t: Optional[int] = None):
        """
            This method is called when a **conflict** is detected.

            :param agents: The other agents who face the conflict. It is *None* when the conflict is caused by an
            obstacle
            :param scenario: Current scenario
            :param t: Current time-step when the conflict is detected.
            :param real_t: Real current time-step. If it is *None*, it will equal to *t*
        """

        ...

    @abstractmethod
    def update(self, scenario: AbstractMap, t: int):
        """
            This method detects any conflicts and fix them.

            :param scenario: Current scenario
            :param t: Current time-step
        """

        ...
