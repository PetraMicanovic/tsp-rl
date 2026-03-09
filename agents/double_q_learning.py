import random
from base_agent import BaseAgent

class DoubleQLearningAgent(BaseAgent):

    def __init__(self, env, alpha, gamma, epsilon):
        super().__init__(env, alpha, gamma, epsilon)

        self.Q2 = {}

    def get_q2_value(self, state, action):
        if state not in self.Q2:
            self.Q2[state] = {}

        if action not in self.Q2[state]:
            self.Q2[state][action] = 0.0

        return self.Q2[state][action]
    
    def update_q2(self, state, action, value):
        if state not in self.Q2:
            self.Q2[state] = {}
        
        self.Q2[state][action] = value

    def best_action_from_q(self, state, actions, q_func):
        best_action = actions[0]
        best_value = q_func(state, best_action)

        for action in actions[1:]:
            value = q_func(state, action)
            if value > best_value:
                best_value = value
                best_action = action
        return best_action
    
    def epsilon_greedy_double(self, state, valid_actions):
        if not valid_actions:
            return None
        
        if random.random() < self.epsilon:
            return random.choice(valid_actions)
        
        best_action = valid_actions[0]
        best_value = self.get_q_value(state, best_action) + self.get_q2_value(state,best_action)

        for action in valid_actions[1:]:
            value = self.get_q_value(state, action) + self.get_q2_value(state, action)

            if value > best_value:
                best_value = value
                best_action = action
        return best_action
    
    def train(self, episodes, num_points = 5):
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

                    if random.random() < 0.5:
                        # update Q1
                        best_next = self.best_action_from_q(next_state, next_valid_actions, self.get_q_value)
                        
                        current_q = self.get_q_value(state, action)
                        next_q = self.get_q2_value(next_state, best_next)

                        new_q = current_q + self.alpha * (reward + self.gamma * next_q - current_q)
                        self.update_q(state, action, new_q)
                    else:
                        # update Q2
                        best_next = self.best_action_from_q(next_state, next_valid_actions, self.get_q2_value)

                        current_q = self.get_q2_value(state, action)
                        next_q = self.get_q_value(next_state, best_next)

                        new_q = current_q + self.alpha * (reward + self.gamma * next_q - current_q)
                        self.update_q2(state, action, new_q)
                    state = next_state
                    valid_actions = next_valid_actions
                else:
                    state = next_state
            rewards_per_episode.append(total_reward)
        return rewards_per_episode