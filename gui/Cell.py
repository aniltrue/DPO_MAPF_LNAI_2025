from typing import Union

import pygame.draw
from gui.DrawableMaterials import DrawableMaterials
from gui.Drawable import *


class Cell(CellDrawable):
    WIDTH: int = 1
    is_observable: bool = False
    material: Union[DrawableMaterials, None] = None

    def draw(self, width: float, height: float, p: float):
        x = self.col * width
        y = self.row * height

        cell_rect = Rect(x, y, width, height)

        pygame.draw.rect(self.surface, "black", cell_rect, self.WIDTH)

        if self.material and (not self.material.partially_observable or self.is_observable):
            self.material.draw(width, height, p)

        if not self.is_observable:
            s = pygame.Surface((width, height))
            s.set_alpha(192)
            s.fill((64, 64, 64))
            self.surface.blit(s, (x, y))
