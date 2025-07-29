import pygame
from pygame import Rect, Surface
from abc import ABC, abstractmethod


class Drawable(ABC):
    surface: Surface

    def __init__(self, surface: Surface):
        self.surface = surface

    @abstractmethod
    def draw(self):
        pass


class CellDrawable(Drawable, ABC):
    row: int
    col: int

    def __init__(self, surface: Surface, row: int, col: int):
        super().__init__(surface)
        self.row = row
        self.col = col

    @abstractmethod
    def draw(self, width: float, height: float, p: float):
        pass
