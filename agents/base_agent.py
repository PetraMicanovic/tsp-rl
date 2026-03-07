import random
from abc import ABC, abstractmethod

class BaseAgent(ABC):
    """
    Base class for reinforcement learning agents. This class provides common functionality for SARSA and Q-learning algorithms.
    """

    def __init__(self, env, alpha, gamma, epsilon):

        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        # Q[state][action] -> Q-value
        self.Q = {}

    def get_state(self):
        visited_tuple = tuple(self.env.visited)

        state = (self.env.current_node, visited_tuple)

        return state
    
    def get_q_value(self, state, action):
        if state not in self.Q:
            self.Q[state] = {}

        if state not in self.Q[state]:
            self.Q[state][action] = 0.0

        return self.Q[state][action]

    def epsilon_greedy(self, state, valid_actions):
        
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
    def train(self, episode):
        pass
        
