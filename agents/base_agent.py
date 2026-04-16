import random
from abc import ABC, abstractmethod
import numpy as np

class BaseAgent(ABC):
    """
    Abstract base class for reinforcement learning agents. This class provides common functionality for SARSA and Q-learning algorithms.
    The class maintains a Q-table and implements:
        - state representation
        - valid action generation
        - epsilon greedy action selection
        - Q access and update
    """

    def __init__(self, env, alpha, gamma, epsilon, epsilon_min, epsilon_decay):
        """
        Initialize the base agent.

        Parameters
        env: TSP environment
            Environment instance the agent interacts with
        alpha: float 
            Learning rate (0 < alpha <= 1)
        gamma: float
            Discount factor for future rewards (0 <= gamma <= 1)
        epsilon: float
            Exploration probability used in epsilon-greedy policy.
        epsilon_min: float
            Minimum value of epsilon after decay. Prevents the agent from becoming fully greedy too early.
        epsilon_decay: float
            Multiplicative decay factor applied to epsilon after each episode. Controls how fast exploration decreases.
        """

        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Q-table structure:
        # Q[state][action] -> Q-value
        self.Q = {}

    def get_state(self):
        """
        Construct the current state representation.

        The state consists of:
            - current_node: index of the node where the agent is currently located
            - visited_mask: binary tuple indicating which intermediate nodes have been visited
            - remaining: number of nodes that still need to be visited
            - dist_to_goal: normalized Euclidean distance from the current node to the goal

        Returns
        state: tuple
            (current_node, visited_mask, remaining, dist_to_goal)
        """
        current_node = self.env.current_node
        max_dist = self.env.max_dist

        visited_mask = []
        for idx in range(self.env.num_points):
            if (idx + 1) in self.env.visited:
                visited_mask.append(1)
            else:
                visited_mask.append(0)

        visited_mask = tuple(visited_mask)

        remaining = self.env.num_points - sum(visited_mask)

        # Distance to goal(normalized to [0,1])
        goal_index = len(self.env.nodes) - 1
        dist_to_goal = self.env._euclidean_distance(current_node, goal_index)

        dist_to_goal = round(dist_to_goal / max_dist, 2)
        
        return (current_node, visited_mask, remaining, dist_to_goal)
            
    def update_q(self, state, action, value):
        """
        Update the Q-table entry for a given state-action pair.

        Parameters
        state: tuple
            State key
        action: int
            Action index
        value: float
            New Q-value
        """
        if state not in self.Q:
            self.Q[state] = {}

        self.Q[state][action] = value

    def get_q_value(self, state, action):
        """
        Retrieve the Q-value for a state-action pair.

        If the state or action does not yet exist in the Q-table, it is initialized with value 0.

        Parameters
        state: tuple
            State key
        action: int
            Action index
        
        Returns
        float
            Q-value for the given state and action
        """
        if state not in self.Q:
            self.Q[state] = {}

        if action not in self.Q[state]:
            self.Q[state][action] = 0.0

        return self.Q[state][action]
    
    def get_valid_actions(self):
        """
        Generate the list of valid actions.

        An action corresponds to selecting one of the intermediate nodes.
        Already visited nodes cannot be selected again.

        Returns
        valid_actions: list[int]
            List of action indices corresponding to unvisited nodes.
        """
        valid_actions = []

        for action in range(self.env.num_points):
            node_index = action + 1

            if node_index not in self.env.visited:
                valid_actions.append(action)

        return valid_actions
    
    def get_combined_q(self, state, action):
        """
        Return the estimated value of a state-action pair.

        For a standard agents(Q-learning, SARSA), this is just the single Q-table value.
        This method exists so that diffrent agents can define their own way of combining value estimates.
        
        Parameters
        state: tuple
            State key
        action: int
            Action index
        
        Returns
        float
            Q-value for the given state and action
        """
        return self.get_q_value(state, action)

    def epsilon_greedy(self, state, valid_actions):
        """
        Select an action using the epsilon-greedy strategy.

        With probability epsilon the agent selects a random valid action.
        Otherwise it selects the action with the highest learned Q-value.

        Parameters
        state: tuple
            Current state
        valid_actions: list[int]
            List of actions that are allowed.

        Returns
        action: int
            Selected action index or None if no valid actions exist
        """
        if not valid_actions:
            return None
        
        # choose a random action
        if random.random() < self.epsilon:
            return random.choice(valid_actions)
        
        # choose the action with the highest Q-value
        best_value = float("-inf")
        best_actions = []

        for action in valid_actions:
            value = self.get_q_value(state, action)
            if value > best_value:
                best_value = value
                best_actions = [action]
            elif value == best_value:
                best_actions.append(action)
        
        return random.choice(best_actions)
    
    @abstractmethod
    def train(self, episodes, num_points = 5):
        """
        Train the agent.

        This method must be implemented by subclasses.

        Parameters
        episodes: int
            Number of training episodes.
        num_points: int
            Number of intermediate nodes used in the environment.

        Returns
        None
        """
        pass