import string
from typing import List
import pygame
from gui.Drawable import *
from gui.Cell import Cell
from gui.DrawableMaterials import *
import numpy as np


class Grid(Drawable):
    WIDTH_RATIO: float = 0.8
    HEIGHT_RATIO: float = 0.7
    grid_x: float
    grid_y: float
    grid_width: float
    grid_height: float
    cells: List[List[Cell]]
    observation_mode: str
    view_x: float = 0
    view_y: float = 0

    def __init__(self, surface: Surface, scenario: np.ndarray, observation_mode: str = "FULL"):
        super().__init__(surface)

        assert observation_mode in ["PARTIAL", "FULL"]

        self.observation_mode = observation_mode

        surface_size = surface.get_size()

        self.grid_x = .1 * surface_size[0]
        self.grid_y = .05 * surface_size[1]
        self.grid_width = self.WIDTH_RATIO * surface_size[0]
        self.grid_height = self.HEIGHT_RATIO * surface_size[1]

        self.s = pygame.Surface((self.grid_width, self.grid_height))

        n = scenario.shape[0]
        self.cells = [[Cell(self.s, i, j) for j in range(n)] for i in range(n)]

    def draw(self, p: float):
        self.s.fill((220, 220, 220))

        pygame.mouse.get_pos()

        w = self.grid_width / len(self.cells)
        h = self.grid_height / len(self.cells)

        for i in range(len(self.cells)):
            for j in range(len(self.cells)):
                self.cells[i][j].draw(w, h, p)

        self.surface.blit(self.s, (self.grid_x, self.grid_y),
                          self.s.get_rect().clip(self.view_x, self.view_y,
                                                 self.WIDTH_RATIO * self.surface.get_size()[0],
                                                 self.HEIGHT_RATIO * self.surface.get_size()[1]))

    def zoom(self, zoom_multiplier: float):
        surface_size = self.surface.get_size()

        self.grid_width = self.WIDTH_RATIO * surface_size[0]
        self.grid_height = self.HEIGHT_RATIO * surface_size[1]

        self.s = pygame.transform.smoothscale(self.s,
                                              (self.grid_width * zoom_multiplier, self.grid_height * zoom_multiplier))

        for i in range(len(self.cells)):
            for j in range(len(self.cells)):
                self.cells[i][j].surface = self.s

        self.grid_width = self.s.get_width()
        self.grid_height = self.s.get_height()

        if zoom_multiplier == 1.:
            self.view_x = 0.
            self.view_y = 0.
        else:
            mouse_pos = pygame.mouse.get_pos()

            self.view_x = mouse_pos[0] - self.grid_x + self.view_x - self.WIDTH_RATIO * surface_size[0] * .5
            self.view_y = mouse_pos[1] - self.grid_y + self.view_y - self.HEIGHT_RATIO * surface_size[1] * .5

        self.move_view(-1)

    def move_view(self, key: int):
        if key == pygame.K_UP:
            self.view_y -= self.grid_height * 0.1
        elif key == pygame.K_DOWN:
            self.view_y += self.grid_height * 0.1
        elif key == pygame.K_LEFT:
            self.view_x -= self.grid_width * 0.1
        elif key == pygame.K_RIGHT:
            self.view_x += self.grid_width * 0.1

        surface_size = self.surface.get_size()

        self.view_x = max(0., min(self.grid_width - self.WIDTH_RATIO * surface_size[0], self.view_x))
        self.view_y = max(0., min(self.grid_height - self.HEIGHT_RATIO * surface_size[1], self.view_y))

    def update(self, scenario: np.ndarray):
        for i in range(len(self.cells)):
            for j in range(len(self.cells)):
                self.cells[i][j].material = None

                if scenario[i, j, 0] == 1:
                    self.cells[i][j].material = DrawableStaticObstacle(self.s, i, j)
                elif scenario.shape[2] > 3 and scenario[i, j, 3] > 0:
                    self.cells[i][j].material = DrawableDynamicObstacle(self.s, i, j, scenario[i, j, 3])
                elif scenario[i, j, 1] > 0:
                    self.cells[i][j].material = DrawableAgent(self.s, i, j, string.ascii_uppercase[scenario[i, j, 1] - 1])
                elif scenario[i, j, 2] > 0:
                    self.cells[i][j].material = DrawableTarget(self.s, i, j, string.ascii_uppercase[scenario[i, j, 2] - 1])

                if self.observation_mode == "PARTIAL":
                    self.cells[i][j].is_observable = scenario[i, j, 4] == 0
                else:
                    self.cells[i][j].is_observable = True

    def on_click(self, mouse_pos: (int, int)) -> (bool, int, int):
        if self.grid_x < mouse_pos[0] < self.grid_x + self.grid_width and self.grid_y < mouse_pos[1] < self.grid_y + self.grid_height:
            x = mouse_pos[0] - self.grid_x + self.view_x
            y = mouse_pos[1] - self.grid_y + self.view_y

            w = self.grid_width / len(self.cells)
            h = self.grid_height / len(self.cells)

            return True, int(y // h), int(x // w)

        return False, -1, -1
