import pickle
import random
import time
import pygame
import numpy as np
from numba import NumbaPendingDeprecationWarning
from gui.Cell import *
from gui.Grid import Grid
from env import update_vision

import warnings
warnings.filterwarnings("ignore", category=NumbaPendingDeprecationWarning)


def generate_map(map_path: str = '', N: int = -1):
    if map_path == '':
        assert N >= 1, "N must be positive!"
        map = np.zeros((N, N, 5), dtype=np.int32)
    else:
        with open(map_path, 'rb') as f:
            map = pickle.load(f)

        N = map.shape[0]

    game = pygame.display.set_mode((1024, 768), 0, 32)
    pygame.font.init()

    clock = pygame.time.Clock()

    zoom_multiplier = 1.

    grid = Grid(game, map, "FULL")
    grid.update(map)

    last_clicked_key = ''
    last_clicked_row_col = (0, 0)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                mouse_pos = pygame.mouse.get_pos()
                clicked, row, col = grid.on_click(mouse_pos)
                if clicked and np.sum(map[row, col, :4]) == 0:
                    if last_clicked_key == 'S':
                        map[row, col, 0] = 1
                        last_clicked_key = 'S'
                    elif last_clicked_key == 'D':
                        map[row, col, 3] = 5
                        last_clicked_row_col = (row, col)
                        last_clicked_key = 'D2'
                    elif last_clicked_key == 'D2':  # UP, DOWN, RIGHT, LEFT
                        last_clicked_key = 'D'
                        if last_clicked_row_col[0] > row:
                            map[last_clicked_row_col[0], last_clicked_row_col[1], 3] = 1
                        elif last_clicked_row_col[0] < row:
                            map[last_clicked_row_col[0], last_clicked_row_col[1], 3] = 2
                        elif last_clicked_row_col[1] > col:
                            map[last_clicked_row_col[0], last_clicked_row_col[1], 3] = 3
                        elif last_clicked_row_col[1] < col:
                            map[last_clicked_row_col[0], last_clicked_row_col[1], 3] = 4
                        else:
                            last_clicked_key = 'D2'
                    elif last_clicked_key == 'A':
                        map[row, col, 1] = np.max(map[:, :, 1]) + 1
                        last_clicked_key = 'A2'
                    elif last_clicked_key == 'A2' or last_clicked_key == 'T':
                        map[row, col, 2] = np.max(map[:, :, 2]) + 1
                        last_clicked_key = 'A'

                    with open(f"map_{N}_{N}.pkl", "wb") as f:
                        pickle.dump(map, f)

            elif event.type == pygame.MOUSEBUTTONUP and event.button == 3:
                mouse_pos = pygame.mouse.get_pos()
                clicked, row, col = grid.on_click(mouse_pos)
                last_clicked_key = ''

                if clicked:
                    map[row, col, :] = 0

                with open(f"map_{N}_{N}.pkl", "wb") as f:
                    pickle.dump(map, f)

            elif event.type == pygame.KEYDOWN:
                c = event.key
                if ord('a') <= c <= ord('z') or ord('A') <= c <= ord('Z') and last_clicked_key != 'D2' and last_clicked_key != 'A2':
                    last_clicked_key = chr(c).upper()
                    if last_clicked_key == 'P':
                        grid.observation_mode = 'PARTIAL'
                        last_clicked_key = ''
                    elif last_clicked_key == 'F':
                        grid.observation_mode = 'FULL'
                        last_clicked_key = ''
                else:
                    grid.move_view(c)

            elif event.type == pygame.MOUSEWHEEL:
                if event.y < 0:
                    zoom_multiplier /= 2.
                else:
                    zoom_multiplier *= 2.

                zoom_multiplier = max(1., min(16, zoom_multiplier))

                grid.zoom(zoom_multiplier)

        game.fill((100, 100, 100))

        update_vision(map)
        grid.update(map)
        grid.draw(1)

        pygame.display.flip()

        clock.tick(60)


if __name__ == "__main__":
    while True:
        print("Welcome,\n\t1) Generate Map\n\t2) Edit Map\n\t3) Exit")

        menu_no = input(" >")

        if menu_no == '1':
            size = int(input("N:"))

            generate_map(N=size)
        if menu_no == '2':
            path = input("Map Path:")

            generate_map(map_path=path)
        elif menu_no == '3':
            print("Bye Bye :)")
            break
        else:
            print("Unknown command!")
