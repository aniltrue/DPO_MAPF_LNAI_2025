"""
    **Initial Path Strategy**: Employs *centralized* algorithm to determine the initial other_path for each agent where only
    *static obstacles* exist and fully observable scenario.
"""

from .InitialPathStrategy import AbstractInitialPathStrategy
from .MAACO1 import MAACO1
from .MAACO2 import MAACO2
from .MAACO0 import MAACO0
from .EECBS import EECBS
from .CBS import CBS
from .LPStrategy import LPStrategy
