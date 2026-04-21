from agents.base_agent import BaseAgent


class QLearningAgent(BaseAgent):
    """
    Q-learning agent implementation.
    """

    def train(self, episodes, num_points=5):
        """
        Train the agent for a specified number of episodes.

        Parameters
        episodes: int
            Number of training episodes.
        num_points: int
            Number of intermediate nodes in the environment.

        Returns
        episode_rewards: list
            Total reward obtained in each episode.
        """
        episode_rewards = []

        # Training loop over episodes
        for episode in range(episodes):
            # Reset environment
            observation, _ = self.env.reset(num_points=num_points)

            state = self.get_state()
            terminated = False
            truncated = False

            total_reward = 0.0

            # Episode loop
            while not (terminated or truncated):

                valid_actions = self.get_valid_actions()

                if not valid_actions:
                    break
                action = self.epsilon_greedy(state, valid_actions)

                # Execute action
                observation, reward, terminated, truncated, _ = self.env.step(action)
                total_reward += reward

                next_state = self.get_state()

                next_valid_actions = self.get_valid_actions()

                # Compute max Q-value for next state

                max_next_q = float("-inf")
                for a in next_valid_actions:
                    q = self.get_q_value(next_state, a)
                    if q > max_next_q:
                        max_next_q = q

                if max_next_q == float("-inf"):
                    max_next_q = 0.0

                current_q = self.get_q_value(state, action)

                # Q-learning update rule
                new_q = current_q + self.alpha * (
                    reward + self.gamma * max_next_q - current_q
                )

                self.update_q(state, action, new_q)
                state = next_state
            episode_rewards.append(total_reward)
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return episode_rewards
