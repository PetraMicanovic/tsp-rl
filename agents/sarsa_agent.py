from base_agent import BaseAgent

class SARSAAgent(BaseAgent):
    """
    This class implements the SARSA(State-Action-Reward-State-Action) on-policy reinforcement learning agent.
    """
    def train(self, episodes, num_points = 5):
        """
        Train the SARSA agent.

        Parameters
        episodes: int
            Number of training episodes.
        num_points: int
            Number of intermediate nodes used in the TSP environment.

        Returns
        rewards_per_episode: list
            Total reward collected in each episode. 
        """
        rewards_per_episode = []

        for episode in range(episodes):
            observation, _ = self.env.reset(num_points = num_points)
            state = self.get_state()
            valid_actions = self.get_valid_actions()

            # Select the first action using the epsilon-greedy policy 
            action = self.epsilon_greedy(state, valid_actions)

            terminated = False
            truncated = False

            total_reward = 0.0

            while not (terminated or truncated):
                # Execute the selected action in the environment
                observation, reward, terminated, truncated, _ = self.env.step(action)

                total_reward += reward

                if not (terminated or truncated):
                    next_state = self.get_state()

                    next_valid_actions = self.get_valid_actions()
                    next_action = self.epsilon_greedy(next_state, next_valid_actions)

                    current_q = self.get_q_value(state, action)
                    next_q = self.get_q_value(next_state, next_action)

                    # Apply the SARSA update rule
                    new_q = current_q + self.alpha * (
                        reward + self.gamma * next_q - current_q
                    )

                    self.update_q(state, action, new_q)
                else:
                    # Update rule for terminal state(no next action) 
                    current_q = self.get_q_value(state, action)
                    new_q = current_q + self.alpha * (reward - current_q)
                    self.update_q(state, action, new_q)
                    next_action = None
                
                state = next_state
                action = next_action

            rewards_per_episode.append(total_reward)
        return rewards_per_episode