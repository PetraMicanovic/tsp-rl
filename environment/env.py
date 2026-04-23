import json
import numpy as np
from gymnasium import spaces


class TSPEnvironment:
    """
    Custom RL environment for the Travelling Salesman navigation problem.

    The environment follows the Gymnasium API with the standard methods:
        - reset()
        - step()
        - render()
        - close()
    The agent starts at a fixed start node and must visit all intermediate nodes before automatically moving to the goal node.
    The objective is to minimize the travelled distance.
    """

    def __init__(self, config_path="config.json"):
        """
        Initializes the environment and loads configuration parameters.

        Parameters
        config_path: str
            Path to the configuration JSON file.
        """

        # Load configuration file
        with open(config_path, "r") as f:
            self.config = json.load(f)

        environment_configuration = self.config["environment"]

        # Start and goal node coordinates
        self.start = np.array(environment_configuration["start_position"], dtype=int)
        self.goal = np.array(environment_configuration["goal_position"], dtype=int)

        # Coordinate limits for sampling intermediate points
        self.x_min = environment_configuration["x_min"]
        self.x_max = environment_configuration["x_max"]
        self.y_min = environment_configuration["y_min"]
        self.y_max = environment_configuration["y_max"]

        self.allowed_points = environment_configuration["num_intermediate_points"]
        self.invalid_action_penalty = self.config["reward_function"][
            "invalid_action_penalty"
        ]

        # Random number generator for reproducibility
        self.seed = environment_configuration["random_seed"]
        self.random_generator = np.random.default_rng(self.seed)

        # Maximum possible distance(diagonal of the workspace)
        self.max_dist = np.sqrt(
            (self.x_max - self.x_min) ** 2 + (self.y_max - self.y_min) ** 2
        )

        # Environment state variables
        self.nodes = None
        self.current_node = None
        self.visited = None
        self.total_distance = None
        self.max_steps = None
        self.steps = None
        self.path = None
        self.episode_reward = None
        self.num_points = None

        self.max_points = max(self.allowed_points)

        # Action space
        # each action corresponds to selecting one intermediate node
        self.action_space = spaces.Discrete(self.max_points)

        # Observation space:
        # [distances_to_nodes, visited_mask]
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(2 * self.max_points,), dtype=np.float32
        )

        self._generate_nodes()

    def _generate_nodes(self):
        """
        Generate a fixed TSP instance(start + intermediate nodes + goal).

        This function is called only once so that the same map is used across all training episodes.
        """
        points = set()
        num_points = self.max_points

        # Generate random intermediate points
        while len(points) < num_points:
            p = tuple(
                self.random_generator.integers(
                    low=[self.x_min, self.y_min],
                    high=[self.x_max + 1, self.y_max + 1],
                )
            )

            if p != tuple(self.start) and p != tuple(self.goal):
                points.add(p)

        points = np.array(list(points), dtype=np.int32)

        # Combine start, intermediate points and goal node
        # 0 -> start
        # 1 .. num_points -> intermediate nodes
        # last-> goal
        self.full_nodes = np.vstack([self.start, points, self.goal])

    def reset(self, num_points=5):
        """
        Starts a new episode.

        The TSP map is generated only once and reused across episodes to ensure stable RL training.
        Parameters
        num_points: int
            Number of intermediate points (5,10,15 or 20)

        Returns:
        observations: np.array
            Initial observation of the environment
        info: dict
            Additional metadata about the episode
        """
        # Validate number of intermediate nodes
        if num_points not in self.allowed_points:
            raise ValueError(f"num_points must be one of {self.allowed_points}")

        self.num_points = num_points

        # Update action space according to the selected number of nodes
        self.action_space = spaces.Discrete(num_points)
        points = self.full_nodes[1:-1][:num_points]

        self.nodes = np.vstack([self.start, points, self.goal])

        # Initial episode state
        self.current_node = 0
        self.visited = {0}
        self.total_distance = 0.0
        self.episode_reward = 0.0

        self.steps = 0
        self.max_steps = len(self.nodes) + self.num_points

        # Path tracking
        self.path = [0]

        observation = self._get_observation()

        info = {"num_nodes": len(self.nodes), "goal_index": len(self.nodes) - 1}

        return observation, info

    def step(self, action):
        """
        Executes one step in the environment.

        Parameters
        action: int
            Index of intermediate node to visit(0 .. num_points-1)

        Returns
        observation: np.array
        reward: float
        terminated: bool
        truncated: bool
        info: dict
        """
        if self.nodes is None:
            raise RuntimeError("Environment must be reset before stepping.")

        terminated = False
        truncated = False

        self.steps += 1

        # Check for invalid action index
        if action < 0 or action >= self.num_points:
            reward = self.invalid_action_penalty
            observation = self._get_observation()
            return observation, reward, False, False, {}

        # Convert action to node index
        node_index = action + 1

        # Check if node was already visited
        if node_index in self.visited:
            reward = self.invalid_action_penalty
            if self.steps >= self.max_steps:
                truncated = True
            return self._get_observation(), reward, False, truncated, {}

        # Primary reward: negative travel distance
        distance = self._euclidean_distance(self.current_node, node_index)
        # normalize by max possible distance
        reward = -(distance / self.max_dist)

        # Update environment state
        self.total_distance += distance
        self.current_node = node_index
        self.visited.add(node_index)
        self.path.append(node_index)

        # reward shaping
        unvisited = []

        for i in range(1, len(self.nodes) - 1):
            if i not in self.visited:
                unvisited.append(i)

        if len(unvisited) > 0:
            nearest = unvisited[0]
            min_distance = self._euclidean_distance(node_index, nearest)

            for i in range(1, len(unvisited)):
                d = self._euclidean_distance(node_index, unvisited[i])
                if d < min_distance:
                    min_distance = d
                    nearest = unvisited[i]

        if self.current_node >= len(self.nodes):
            raise ValueError(f"Invalid current_node: {self.current_node}")

        # Termination condition
        # Check if all intermediate nodes have been visited
        if len(self.visited) == len(self.nodes) - 1:
            goal_index = len(self.nodes) - 1
            goal_distance = self._euclidean_distance(self.current_node, goal_index)

            self.total_distance += goal_distance
            reward += -goal_distance / self.max_dist

            # Large bonus for completing the tour
            normalized_total = self.total_distance / (self.num_points * self.max_dist)
            reward += 5.0 / (1.0 + normalized_total)

            self.current_node = goal_index
            self.path.append(goal_index)

            terminated = True

        # Truncation condition
        if self.steps >= self.max_steps and not terminated:
            truncated = True

        observation = self._get_observation()
        self.episode_reward += reward

        info = {
            "total_distance": self.total_distance,
            "visited_count": len(self.visited),
            "incomplete_tour": len(self.visited) != len(self.nodes) - 1,
        }

        return observation, reward, terminated, truncated, info

    def _get_observation(self):
        """
        Construct the observation vector.

        Observation = [distances, visited_mask]
        distances: Euclidean distances from current node to each intermediate node.
        visited_mask: binary indicator(1 if visited, 0 otherwise)
        """
        if self.current_node >= len(self.nodes):
            raise ValueError(f"Invalid current_node: {self.current_node}")

        distances = np.zeros(self.max_points, dtype=np.float32)
        visited_mask = np.zeros(self.max_points, dtype=np.float32)

        # goal node excluded from observation because it is reached automatically
        for idx in range(self.num_points):
            node_index = idx + 1
            if node_index in self.visited:
                distances[idx] = 0.0
                visited_mask[idx] = 1.0
            else:
                # Normalize distances to [0,1] for more stable learning
                distances[idx] = (
                    self._euclidean_distance(self.current_node, node_index)
                    / self.max_dist
                )
                visited_mask[idx] = 0.0
        observation = np.concatenate((distances, visited_mask))

        return observation

    def _euclidean_distance(self, i, j):
        """
        Compute Euclidean distance between nodes i and j.
        """
        if i >= len(self.nodes) or j >= len(self.nodes):
            raise IndexError(f"Invalid indices i={i}, j={j}, len={len(self.nodes)}")
        return np.sqrt(np.sum((self.nodes[i] - self.nodes[j]) ** 2))

    def render(self):
        """
        Print current state of the environment.
        """
        print("\n-----ENVIRONMENT STATE------")
        print("Current node: ", self.current_node)
        print("Current position:", self.nodes[self.current_node])
        print("Visited nodes: ", self.visited)
        print("Total distance: ", self.total_distance)
        print("Path: ", self.path)

    def close(self):
        """
        Placeholder for compatibility with Gymnasium API.
        """
        pass
