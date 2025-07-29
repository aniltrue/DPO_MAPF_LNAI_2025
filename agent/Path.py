from typing import TypeVar, List, Tuple

Coordinate = TypeVar("Coordinate", bound=Tuple[int, int])
Path = TypeVar("Path", bound=List[Tuple[int, int]])
