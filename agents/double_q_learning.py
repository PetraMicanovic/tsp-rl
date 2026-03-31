import random
from agents.base_agent import BaseAgent

class DoubleQLearningAgent(BaseAgent):
    """
    Double Q-Learning agent maintains two separate Q-tables (Q1 and Q2) in order to reduce overestimation bias that appears in standard Q-learning.
    At every step of training:
        - one of the Q-tables is randomly selected for update
        - the other Q-table is used to evaluate the next action
    """
    def __init__(self, env, alpha, gamma, epsilon, epsilon_min, epsilon_decay):
        """
        Initialize the Double Q-Learning agent.

        Parameters
        env: environment
        alpha: float
            Learning rate
        gamma: float
            Discount factor for future rewards
        epsilon: float
            Exploration probability used in epsilon-greedy policy.
        epsilon_min: float
            Minimum value of epsilon after decay. Prevents the agent from becoming fully greedy too early.
        epsilon_decay: float
            Multiplicative decay factor applied to epsilon after each episode. Controls how fast exploration decreases.
        """
        super().__init__(env, alpha, gamma, epsilon, epsilon_min, epsilon_decay)

        self.Q2 = {}

    def get_q2_value(self, state, action):
        """
        Return the Q2 value for a given (state, action) pair.
        If the state or action does not yet exist in the Q2 table, it is initialized with value 0.0.

        Parameters
        state: tuple
            Current environment state
        action: int
            Action taken from the state

        Returns
        float
            Q2 value for the given state-action pair.
        """
        if state not in self.Q2:
            self.Q2[state] = {}

        if action not in self.Q2[state]:
            self.Q2[state][action] = 0.0

        return self.Q2[state][action]
    
    def update_q2(self, state, action, value):
        """
        Update Q2 value for a specific (state, action).

        Parameters
        state: tuple
            Current environment state
        action: int
            Action taken from the state
        value: float
            New Q value to assign
        """
        if state not in self.Q2:
            self.Q2[state] = {}
        
        self.Q2[state][action] = value

    def get_combined_q(self, state, action):
        """
        Return combined Q-value estmate for Double Q-learning.
        """
        return self.get_q_value(state, action) + self.get_q2_value(state, action)

    def best_action_from_q(self, state, actions, q_func):
        """
        Find the action with the highest Q-value according to a given Q-function.
        This helper function is used to select the best action either from Q1 or Q2 depending on which table is used for evaluation.

        Parameters
        state: tuple
            Current environment state
        actions: list
            List of valid actions
        q_func: callable
            Function that returns Q-value(e.g. get_q_value or get_q2_value)
        
        Returns
        best_action: int
            Action with the highest Q-value
        """
        best_action = actions[0]
        best_value = q_func(state, best_action)

        # Iterate through remaining actions and find the maximum Q-value
        for action in actions[1:]:
            value = q_func(state, action)
            if value > best_value:
                best_value = value
                best_action = action
        return best_action
    
    def epsilon_greedy_double(self, state, valid_actions):
        """
        Select an action using epsilon-greedy policy.
        With probability:
            - epsilon -> choose a random action
            - (1 - epsilon) -> choose the action with the highest value according to the combined estimate (Q1+Q2)
        Using the sum of Q1 and Q2 provides a better estimate of the true action value.

        Parameters
        state: tuple
            Current environment state
        valid_actions: list
            List of valid actions in the current state

        Returns
        action 
            Selected action
        """
        if not valid_actions:
            return None
        
        if random.random() < self.epsilon:
            return random.choice(valid_actions)
        
        # Randomly choose which Q-table to use for action selection
        if random.random() < 0.5:
            q_func = self.get_q_value
        else:
            q_func = self.get_q2_value
        
        best_action = valid_actions[0]
        best_value = q_func(state, best_action)

        for action in valid_actions[1:]:
            value = q_func(state, action)
            if value > best_value:
                best_value = value
                best_action = action
        return best_action
    
    def train(self, episodes, num_points = 5):
        """
        Train the agent using Double Q-Learning.

        Parameters
        episodes: int
            Number of training episodes
        num_points: int
            Number of points used in the TSP environment

        Returns
        rewards_per_episode: list
            Total reward collected in each episode.
        """
        rewards_per_episode = []

        for episode in range(episodes):
            observation, _ = self.env.reset(num_points = num_points)
            state = self.get_state()

            terminated = False
            truncated = False
            total_reward = 0.0

            valid_actions = self.get_valid_actions()

            while not(terminated or truncated):
                action = self.epsilon_greedy_double(state, valid_actions)
                observation, reward, terminated, truncated, _ = self.env.step(action)
                next_state = self.get_state()
                total_reward += reward

                if not (terminated or truncated):
                    next_valid_actions = self.get_valid_actions()

                    # Randomly decide which Q-table to update
                    if random.random() < 0.5:
                        # update Q1
                        if next_valid_actions:
                            best_next = self.best_action_from_q(next_state, next_valid_actions, self.get_q_value)
                            next_q = self.get_q2_value(next_state, best_next)
                        else:
                            next_q = 0

                        current_q = self.get_q_value(state, action)
                        # Double Q-learning update rule
                        new_q = current_q + self.alpha * (reward + self.gamma * next_q - current_q)
                        self.update_q(state, action, new_q)
                    else:
                        if next_valid_actions:
                            # update Q2
                            best_next = self.best_action_from_q(next_state, next_valid_actions, self.get_q2_value)
                            next_q = self.get_q_value(next_state, best_next)
                        else:
                            next_q = 0

                        current_q = self.get_q2_value(state, action)
                        new_q = current_q + self.alpha * (reward + self.gamma * next_q - current_q)
                        self.update_q2(state, action, new_q)
                    state = next_state
                    valid_actions = next_valid_actions
                else:
                    # terminal update
                    if random.random() < 0.5:
                        current_q = self.get_q_value(state, action)
                        new_q = current_q + self.alpha * (reward - current_q)
                        self.update_q(state, action, new_q)
                    else: 
                        current_q = self.get_q2_value(state, action)
                        new_q = current_q + self.alpha * (reward  - current_q)
                        self.update_q2(state, action, new_q)
                    break

            rewards_per_episode.append(total_reward)
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return rewards_per_episode