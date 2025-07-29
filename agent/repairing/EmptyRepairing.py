from typing import Optional, List

from env.map import AbstractMap
from .RepairingStrategy import AbstractRepairingStrategy
from .. import Path, AbstractAgent, Coordinate


class EmptyRepairing(AbstractRepairingStrategy):
    """
        This repairing strategy does not repair.
    """
    def __init__(self, max_iter: int = 150, initial_b: float = 5.0, end_b: float = 0.5, number_of_ants: int = 75):
        super().__init__()

    @property
    def name(self) -> str:
        return "No-Repairing"

    def repair(self, path: Path, agent: AbstractAgent, t: int, real_t: int, map: AbstractMap,
               other_paths: Optional[List[Path]]) -> Optional[Path]:

        return path