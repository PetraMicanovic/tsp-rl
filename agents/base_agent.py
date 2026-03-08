import random
from abc import ABC, abstractmethod

class BaseAgent(ABC):
    """
    Abstract base class for reinforcement learning agents. This class provides common functionality for SARSA and Q-learning algorithms.
    The class maintains a Q-table and implements:
        - state representation
        - valid action generation
        - epsilon greedy action selection
        - Q access and update
    """

    def __init__(self, env, alpha, gamma, epsilon):
        """
        Initialize the base agent.

        Parameters
        env: TSP environment
            Environment instance the agent interacts with
        alpha: float 
            Learning rate (0 < alpha <= 1)
        gamma: float
            Discount factor for future rewards (0 <= alpha <= 1)
        epsilon: float
            Exploration probability used in epsilon-greedy policy.
        """

        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        # Q-table structure:
        # Q[state][action] -> Q-value
        self.Q = {}

    def get_state(self):
        """
        Construct the current state representation.

        The state consist of:
            - current node where the agent is located
            - tuple of visited nodes
        The visisted nodes are sorted to ensure that the same set of visited nodes always produces the same state key.

        Returns
        state: tuple
            (current_node, visited_nodes_tuple)
        """
        visited_tuple = tuple(sorted(self.env.visited))

        state = (self.env.current_node, visited_tuple)

        return state
    
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
            List of action indices that correspond to unvisited nodes.
        """
        valid_actions = []

        for action in range(self.env.num_points):
            node_index = action + 1

            if node_index not in self.env.visited:
                valid_actions.append(action)

        return valid_actions

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
        int or None
            Selected action index or None if no valid actions exist
        """
        if not valid_actions:
            return None
        
        # choose a random action
        if random.random() < self.epsilon:
            return random.choice(valid_actions)
        
        # choose the action with the highest Q-value
        best_action = valid_actions[0]
        best_value = self.get_q_value(state,best_action)

        for action in valid_actions:
            value = self.get_q_value(state, action)

            if value > best_value:
                best_value = value
                best_action = action
        
        return best_action
    
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
        
