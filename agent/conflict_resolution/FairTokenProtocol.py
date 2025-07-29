from threading import Thread
from typing import Dict, Optional
from env.map import AbstractMap
from .PriorityProtocol import AbstractPriorityProtocol
from agent.AbstractAgent import AgentID
from agent.Path import Path, Coordinate
from ..utils import find_current_and_goal_points


class FairTokenProtocol(AbstractPriorityProtocol):
    """
        This *protocol* prioritizes agents based on the urgency and the number of tokens

            - If the agent has urgency, it will be prioritized.
            - Else if the agent has the lowest number of tokens, it will be prioritized.
    """

    @property
    def name(self) -> str:
        return "Fair Token Protocol"

    def update(self, prioritized_agent, other_agents: list, repaired_paths: Dict[AgentID, Path], t: int):
        """
            This method updates the paths and **tokens** for the agents

            :param prioritized_agent: The prioritized agent who will not change its route
            :param other_agents: List of other agents who will change their routes
            :param repaired_paths: List of new paths
            :param t: Current time-step
        """
        # Decrease the number of tokens of the prioritized agent
        self.token_map[prioritized_agent.id] = self.token_map.get(prioritized_agent.id, 1) - 1

        # Update the paths and increase the number of tokens for the other agents
        for other_agent in other_agents:
            self.token_map[other_agent.id] = self.token_map.get(other_agent.id, 1) + 1

            if repaired_paths[other_agent.id] is not None:
                other_agent.path = repaired_paths[other_agent.id]

    def resolve(self, agents: list, scenario: AbstractMap, t: int, real_t: Optional[int] = None):

        if real_t is None:
            real_t = t

        # Find repaired paths
        repaired_paths: Dict[AgentID, Path] = {}

        # Multi-Threaded Fashion
        threads = []

        def repair(target_agent):
            other_paths = [path for agent_id, path in target_agent.other_agents_path.items() if agent_id != target_agent.id]

            repaired_path = self.repairing_strategy.repair(target_agent.path, target_agent, t, real_t, scenario,
                                                           other_paths)

            repaired_paths[target_agent.id] = repaired_path

        for agent in agents:
            thread = Thread(target=repair, args=(agent,))
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()

        # Check for urgent
        urgent_agent_id = -1
        for agent in agents:
            if agent.id not in self.token_map:
                self.token_map[agent.id] = 1

            if repaired_paths[agent.id] is None:  # If ACO cannot find any repaired other_path
                urgent_agent_id = agent.id

                break

            if agent.t_max < len(repaired_paths[agent.id]) - 1:  # Prioritize if remaining time is not enough
                urgent_agent_id = agent.id

                break

            (current_i, current_j), (goal_i, goal_j) = find_current_and_goal_points(scenario.get_data(), agent.id)

            if current_i == goal_i and current_j == goal_j: # Check for goal point
                urgent_agent_id = agent.id

                break

        if urgent_agent_id != -1:  # Is there any urgent one
            prioritized_agent = [agent for agent in agents if agent.id == urgent_agent_id][0]
            other_agents = [agent for agent in agents if agent.id != urgent_agent_id]

            self.update(prioritized_agent, other_agents, repaired_paths, t)

            return

        # Check for token
        sorted_agent_ids = sorted([agent.id for agent in agents], key=lambda agent_id: self.token_map[agent_id],
                                  reverse=True)

        if self.token_map[sorted_agent_ids[0]] > self.token_map[sorted_agent_ids[1]]:
            prioritized_agent_id = sorted_agent_ids[0]
            prioritized_agent = [agent for agent in agents if agent.id == prioritized_agent_id][0]
            other_agents = [agent for agent in agents if agent.id != prioritized_agent_id]

            self.update(prioritized_agent, other_agents, repaired_paths, t)

            return

        # Check for distance difference
        sorted_agent = sorted([agent for agent in agents],
                              key=lambda a: abs(len(a.path) - len(repaired_paths[a.id])),
                              reverse=True)

        self.update(sorted_agent[0], sorted_agent[1:], repaired_paths, t)
