from typing import Union
import pygame
from pygame import Surface, Rect
from gui.Drawable import CellDrawable
from abc import ABC


class DrawableMaterials(CellDrawable, ABC):
    color: (int, int, int)
    partially_observable: bool
    text: Union[str, None]
    is_circled: bool

    def __init__(self, surface: Surface, color: (int, int, int), row: int, col: int, is_circled: bool = True,
                 partially_observable: bool = False, text: Union[str, None] = None):
        super().__init__(surface, 0, 0)

        self.color = color
        self.partially_observable = partially_observable
        self.text = text
        self.is_circled = is_circled

        self.row = row
        self.col = col

    def draw(self, width: float, height: float, p: float):
        x = self.col * width
        y = self.row * height

        if self.partially_observable:
            pass
            """
            if self.text == 'Up':
                y -= (p - 1) * height * .5
            elif self.text == 'Down':
                y += (p - 1) * height * .5
            elif self.text == 'Left':
                x -= (p - 1) * width * .5
            elif self.text == 'Right':
                x += (p - 1) * width * .5

            x = max(0., x)
            y = max(0., y)
            """

        cell_rect = Rect(x, y, width, height)

        if self.is_circled:
            pygame.draw.ellipse(self.surface, self.color, cell_rect)
        else:
            pygame.draw.rect(self.surface, self.color, cell_rect)

        if self.text:
            my_font = pygame.font.SysFont('Comic Sans MS', round(3 * width / 10))
            text_surface = my_font.render(self.text, True, (255, 255, 255))
            self.surface.blit(text_surface, (x + width * 0.15, y + height * 0.05))


class DrawableStaticObstacle(DrawableMaterials):
    def __init__(self, surface: Surface, row: int, col: int):
        super().__init__(surface, (0, 0, 0), row, col, is_circled=False)


class DrawableDynamicObstacle(DrawableMaterials):
    def __init__(self, surface: Surface, row: int, col: int, value: int):
        text = ['Up', 'Down', 'Left', 'Right', '']
        super().__init__(surface, (200, 0, 0), row, col, partially_observable=True, text=text[value - 1])


class DrawableAgent(DrawableMaterials):
    def __init__(self, surface: Surface, row: int, col: int, agent_no: str):
        super().__init__(surface, (0, 0, 200), row, col, text=f"A({agent_no})")


class DrawableTarget(DrawableMaterials):
    def __init__(self, surface: Surface, row: int, col: int, agent_no: str):
        super().__init__(surface, (0, 200, 0), row, col, text=f"T({agent_no})")
