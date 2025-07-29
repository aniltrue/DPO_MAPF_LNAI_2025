import threading
import time
from threading import Thread
from typing import List, Dict, Any, Optional, Union, Tuple

import numpy as np
import pygame
from scipy.optimize import linear_sum_assignment

from agent import Path
from agent.utils import find_current_and_goal_points
from gui import Grid
from .map.Map import AbstractMap
from agent.AbstractAgent import AbstractAgent, AgentID


class Environment:
    """
        This class simulates the MAPF scenario
    """

    agents: List[AbstractAgent]             #: List of agents
    map: AbstractMap                        #: Scenario
    will_draw: bool                         #: Whether the environment will be drawn, or not
    draw_thread: Optional[Thread]           #: PyGame thread
    goals: Dict[AgentID, Tuple[int, int]]   #: Goal positions
    initial_paths: Dict[AgentID, Path]      #: Initial agents' paths in the beginning of the simulation

    def __init__(self, map: AbstractMap, agents: List[AbstractAgent], will_draw: bool = False):
        self.map = map
        self.agents = agents
        self.will_draw = will_draw
        self.goals = {}
        self.initial_paths = {}

        if self.will_draw:
            self.thread = threading.Thread(target=self.draw)
            self.thread.daemon = True
            self.thread.start()
        else:
            self.thread = None

    def draw(self):
        """
            This method draw the environment
        """
        
        # Initiate
        pygame.font.init()

        surface = pygame.display.set_mode((1024, 768), 0, 32)
        grid = Grid(surface, self.map.get_raw_data())

        surface.fill((100, 100, 100))

        grid.update(self.map.get_raw_data())

        grid.draw(1.)

        pygame.display.flip()

        zoom_multiplier = 1.

        clock = pygame.time.Clock()

        # Main loop
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

                elif event.type == pygame.KEYDOWN:
                    c = event.key
                    if ord('a') <= c <= ord('z') or ord('A') <= c <= ord('Z'):
                        key = chr(c).upper()
                        if key == 'P':
                            grid.observation_mode = 'PARTIAL'
                        elif key == 'F':
                            grid.observation_mode = 'FULL'
                    else:
                        grid.move_view(c)

                elif event.type == pygame.MOUSEWHEEL:
                    if event.y < 0:
                        zoom_multiplier /= 2.
                    else:
                        zoom_multiplier *= 2.
                        
                    zoom_multiplier = max(1., min(16, zoom_multiplier))
                    
                    grid.zoom(zoom_multiplier)
                    
            surface.fill((100, 100, 100))

            grid.update(self.map.get_raw_data())

            grid.draw(1.0)

            pygame.display.flip()

            clock.tick(120)

    def collision_check(self, t: int) -> Dict[str, Union[bool, str]]:
        """
            This method checks any agent-based collision at time-step *t*

            :param t: Target time step
            :return: Feasibility with message
        """

        cells_in_usage = set()
        agent_moves = set()

        for agent in self.agents:
            if len(agent.path) <= 0:
                return {"feasibility": False, "result": "END_OF_PATH"}

            if len(agent.path) <= t:
                (current_i, current_j) = agent.path[-1]
            else:
                (current_i, current_j) = agent.path[t]

            (goal_i, goal_j) = self.goals[agent.id]

            if current_i == goal_i and current_j == goal_j:
                continue

            if t > agent.t_max:
                return {"feasibility": False, "result": "END_OF_CAPACITY"}

            if len(agent.path) <= t + 1:  # If no more action
                if t > 0:
                    while len(agent.path) <= t + 1:  # Add WAIT actions
                        agent.path.append(agent.path[-1])
                else:
                    return {"feasibility": False, "result": "END_OF_PATH"}

            next_i, next_j = agent.path[t + 1]

            if abs(next_i - current_i) + abs(next_j - current_j) > 1:  # Invalid move
                return {"feasibility": False, "result": "INVALID_MOVE"}

            if (next_i, next_j) in cells_in_usage:
                return {"feasibility": False, "result": "AGENT_COLLISION"}

            cells_in_usage.add((next_i, next_j))

            if (next_i, next_j, current_i, current_j) in agent_moves:  # Swapping
                return {"feasibility": False, "result": "AGENT_COLLISION"}

            agent_moves.add((current_i, current_j, next_i, next_j))

        return {"feasibility": True, "result": ""}

    def run(self) -> Dict[str, Any]:
        """
            This method runs the given scenario

            :return: Result as a dictionary
        """

        lower_bound: float = 0.

        for agent in self.agents:
            (start_i, start_j), (goal_i, goal_j) = find_current_and_goal_points(self.map.get_data(), agent.id)

            # Remember all goals
            self.goals[agent.id] = (goal_i, goal_j)

            # Calculate lower bound
            lower_bound += abs(start_i - goal_i) + abs(start_j - goal_j)

        cost_max = sum([agent.t_max for agent in self.agents])

        self.map.initiate()  # Initiate scenario
        t = 0
        cost = 0

        # Store the initial paths for Average EMD calculation
        self.initial_paths = {agent.id: agent.path.copy() for agent in self.agents}

        start_time = time.time()

        while True:  # Main loop
            if self.will_draw:  # Wait for drawing
                time.sleep(0.1)

            finish_count = 0  # Number of agent who reaches the goal

            for agent in self.agents:
                # Collision detection + Repairing
                agent.update(self.map, t)

            collision_report = self.collision_check(t)

            if not collision_report["feasibility"]:
                if self.will_draw:
                    time.sleep(0.1)
                    self.map.initiate()

                return {"feasibility": False,
                        "result": collision_report["result"],
                        "time_step": t,
                        "elapsed_time": time.time() - start_time,
                        "cost": cost,
                        "normalized_cost": cost / cost_max,
                        "max_token_diff": self.agents[0].priority_protocol.get_max_difference(
                            [agent.id for agent in self.agents]),
                        "lower_bound": lower_bound,
                        "average_emd": self.calculate_emd(),
                        "finish_count": finish_count}

            for agent in self.agents:  # From
                (goal_i, goal_j) = self.goals[agent.id]

                if len(agent.path) <= t:
                    (current_i, current_j) = agent.path[-1]
                else:
                    (current_i, current_j) = agent.path[t]

                if current_i == goal_i and current_j == goal_j:  # Check whether the agent reaches its goal, or not
                    finish_count += 1

                    continue

                self.map[current_i, current_j, 1] = 0

            for agent in self.agents:  # To
                (goal_i, goal_j) = self.goals[agent.id]

                if len(agent.path) <= t:
                    (current_i, current_j) = agent.path[-1]
                else:
                    (current_i, current_j) = agent.path[t]

                if current_i == goal_i and current_j == goal_j:  # Check whether the agent reaches its goal, or not
                    continue

                # Get next destination
                next_i, next_j = agent.path[t + 1]

                if next_i != goal_i or next_j != goal_j:  # Remove when reaching the goal point
                    self.map[next_i, next_j, 1] = agent.id
                else:
                    self.map[goal_i, goal_j, 2] = 0

                # Increase the cost
                cost += 1

            if self.will_draw:  # Wait for drawing after the agent move
                time.sleep(0.1)

            if finish_count == len(self.agents):  # If the system is completed
                if self.will_draw:  # Update vision
                    self.map.initiate()

                return {"feasibility": True,
                        "result": "FINISH",
                        "time_step": t,
                        "elapsed_time": time.time() - start_time,
                        "cost": cost,
                        "normalized_cost": cost / cost_max,
                        "max_token_diff": self.agents[0].priority_protocol.get_max_difference(
                            [agent.id for agent in self.agents]),
                        "lower_bound": lower_bound,
                        "average_emd": self.calculate_emd(),
                        "finish_count": finish_count}

            # Check for obstacle-agent collision before the obstacles move
            if self.map.check_collision():
                if self.will_draw:  # Update vision
                    self.map.initiate()

                return {"feasibility": False,
                        "result": "OBSTACLE_COLLISION",
                        "time_step": t,
                        "elapsed_time": time.time() - start_time,
                        "cost": cost,
                        "normalized_cost": cost / cost_max,
                        "max_token_diff": self.agents[0].priority_protocol.get_max_difference(
                            [agent.id for agent in self.agents]),
                        "lower_bound": lower_bound,
                        "average_emd": self.calculate_emd(),
                        "finish_count": finish_count}

            # Update vision + Move dynamic obstacle
            self.map.step()

            if self.will_draw:  # Wait for drawing after the obstacles move
                time.sleep(0.1)

            # Check for obstacle-agent collision after the dynamic obstacles move
            if self.map.check_collision():
                if self.will_draw:  # Update vision
                    self.map.initiate()

                return {"feasibility": False,
                        "result": "OBSTACLE_COLLISION",
                        "time_step": t,
                        "elapsed_time": time.time() - start_time,
                        "cost": cost,
                        "normalized_cost": cost / cost_max,
                        "max_token_diff": self.agents[0].priority_protocol.get_max_difference(
                            [agent.id for agent in self.agents]),
                        "lower_bound": lower_bound,
                        "average_emd": self.calculate_emd(),
                        "finish_count": finish_count}

            t += 1

            if self.will_draw:  # Wait for drawing for the next time-step
                time.sleep(0.1)


    def calculate_emd(self) -> float:
        """
            This method calculates the *Earth Moving Distance* between the first and last paths for each agent.
            Then, it provides the average EMD value.

        :return: Average EMD between the first and last paths.
        """

        total_emd = 0.
        counter = 0

        for agent in self.agents:
            total_emd += self.emd(self.initial_paths[agent.id], agent.path, self.goals[agent.id])
            counter += 1

        return total_emd / counter

    @staticmethod
    def emd(path1: Path, path2: Path, dummy_cell: Optional[Tuple[int, int]] = None) -> float:
        """
            This method calculates *Earth Moving Distance* between two given paths.

        :param path1: First path
        :param path2: Second path
        :param dummy_cell: Dummy cell (e.g., Goal cell)
        :return: EMD between two paths
        """
        p1 = path1.copy()
        p2 = path2.copy()

        # Make their lengths equal by appending *WAITING* action on the dummy cell or goal cell.
        while len(p1) < len(p2):
            p1.append(dummy_cell if dummy_cell is not None else p2[-1].copy())

        while len(p2) < len(p1):
            p2.append(dummy_cell if dummy_cell is not None else p1[-1].copy())

        # Create the cost matrix (distance matrix)
        n = len(p1)
        m = len(p2)

        cost_matrix = np.zeros((n, m))

        # Fill in the cost matrix with the Manhattan distances between each pair of points
        for i in range(n):
            for j in range(m):
                cost_matrix[i, j] = abs(p1[i][0] - p2[j][0]) + abs(p1[i][1] - p2[j][1])

        # Solve the linear sum assignment problem (Hungarian algorithm)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # Calculate the total cost (sum of the optimal assignment)
        total_cost = cost_matrix[row_ind, col_ind].sum()

        return float(total_cost)
