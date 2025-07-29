import random
from typing import Optional

from env.map import AbstractMap
from .PriorityProtocol import AbstractPriorityProtocol
from agent.Path import Coordinate


class RandomProtocol(AbstractPriorityProtocol):
    """
        This protocol randomly prioritize the agents
    """

    @property
    def name(self) -> str:
        return "Random Protocol"

    def resolve(self, agents: list, scenario: AbstractMap, t: int, real_t: Optional[int] = None):

        if real_t is None:
            real_t = t

        prioritized_one = random.choice(list(range(len(agents))))  # Randomly select a prioritized agent

        prioritized_agent_id = agents[prioritized_one].id

        agents.pop(prioritized_one)

        # Update tokens

        self.token_map[prioritized_agent_id] = self.token_map.get(prioritized_agent_id, 1) - 1

        for agent in agents:
            self.token_map[agent.id] = self.token_map.get(agent.id, 1) + 1

        # Update paths
        for agent in agents:  # Update others
            other_paths = [path for agent_id, path in agent.other_agents_path.items() if agent_id != agent.id]

            fixed_path = self.repairing_strategy.repair(agent.path, agent, t, real_t, scenario, other_paths)

            if fixed_path is None:  # Wait if it cannot be fixed
                agent.path.append(agent.path[-1])

                return

            agent.path = fixed_path
