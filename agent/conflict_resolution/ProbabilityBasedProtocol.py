import random
from typing import List, Optional
from env.map import AbstractMap
from .PriorityProtocol import AbstractPriorityProtocol
from agent.Path import Coordinate
from agent.utils import find_current_and_goal_points


class ProbabilityBasedProtocol(AbstractPriorityProtocol):
    @property
    def name(self) -> str:
        return "Probability-Based Protocol"

    def resolve(self, agents: list, scenario: AbstractMap, t: int, real_t: Optional[int] = None):
        if real_t is None:
            real_t = t

        map_data = scenario.get_data()

        # Calculate remaining capacities
        remaining_capacities = [max(agent.t_max - t, 1) for agent in agents]

        # Calculate remaining distances
        remaining_distances = []

        for agent in agents:
            (current_i, current_j), (goal_i, goal_j) = find_current_and_goal_points(map_data, agent.id)

            remaining_distances.append(abs(current_i - goal_i) + abs(current_j - goal_j) + 1)

        # Calculate priority scores
        priority_scores = [max(remaining_capacities[i] - remaining_distances[i], 0.) for i in range(len(agents))]
        priority_scores = [priority_scores[i] / (sum(priority_scores) + 1e-12) for i in range(len(agents))]

        # Normalize
        remaining_distances = [dist / (sum(remaining_distances) + 1e-12) for dist in remaining_distances]
        remaining_capacities = [capacity / (sum(remaining_capacities) + 1e-12) for capacity in remaining_capacities]

        # Calculate risk factors
        risk_factors = [remaining_distances[i] * (1. - remaining_capacities[i]) for i in range(len(agents))]

        # Concede probability
        probabilities = [risk_factors[i] * priority_scores[i] for i in range(len(agents))]
        probabilities = [probability / (sum(probabilities) + 1e-12) for probability in probabilities]

        # Select agent
        selected_agent = agents[self.random_selection(probabilities)]
        other_agents = [agent for agent in agents if agent.id != selected_agent.id]

        # Update tokens
        self.token_map[selected_agent.id] = self.token_map.get(selected_agent.id, 1) - 1

        for agent in other_agents:
            self.token_map[agent.id] = self.token_map.get(agent.id, 1) + 1

        # Update other agents
        for agent in other_agents:
            other_paths = [path for agent_id, path in agent.other_agents_path.items() if agent_id != agent.id]

            fixed_path = self.repairing_strategy.repair(agent.path, agent, t, real_t, scenario, other_paths)

            if fixed_path is not None:
                agent.path = fixed_path

    @staticmethod
    def random_selection(probabilities: List[float]) -> int:
        """
            This method randomly select an index based on the given probabilities

            :param probabilities: Given probability
            :return: Randomly selected index
        """
        p = 0
        r = random.random()

        for i in range(len(probabilities)):
            if p <= r < p + probabilities[i]:
                return i

            p += probabilities[i]

        return -1
