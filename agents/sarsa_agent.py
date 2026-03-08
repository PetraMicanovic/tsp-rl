from base_agent import BaseAgent

class SARSAAgent(BaseAgent):
    def train(self, episodes, num_points = 5):
        rewards_per_episode = []

        for episode in range(episodes):
            observation, _ = self.env.reset(num_points = num_points)
            state = self.get_state()
            valid_actions = self.get_valid_actions()

            action = self.epsilon_greedy(state, valid_actions)

            terminated = False
            truncated = False

            total_reward = 0.0

            while not (terminated or truncated):
                observation, reward, terminated, truncated, _ = self.env.step(action)

                next_state = self.get_state()
                total_reward += reward

                if not (terminated or truncated):
                    next_valid_actions = self.get_valid_actions()
                    next_action = self.epsilon_greedy(next_state, next_valid_actions)

                    current_q = self.get_q_value(state, action)
                    next_q = self.get_q_value(next_state, next_action)

                    new_q = current_q + self.alpha * (
                        reward + self.gamma * next_q - current_q
                    )

                    self.update_q(state, action, new_q)
                else: 
                    current_q = self.get_q_value(state, action)
                    new_q = current_q + self.alpha * (reward - current_q)
                    self.update_q(state, action, new_q)
                    next_action = None
                
                state = next_state
                action = next_action

            rewards_per_episode.append(total_reward)
        return rewards_per_episode