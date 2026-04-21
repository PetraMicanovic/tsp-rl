from agents.base_agent import BaseAgent


class NStepSARSAAgent(BaseAgent):
    """
    The n-step SARSA algorithm extends the standard SARSA method by using multiple future rewards (n steps) when updating Q-values.
    The update uses accumulated rewards over the next n steps.
    """

    def __init__(self, env, alpha, gamma, epsilon, epsilon_min, epsilon_decay, n=10):
        """
        Initialize the n-step SARSA agent.

        Parameters
        env: environment
            Environment instance the agent interacts with.
        alpha: float
            Learning rate
        gamma: float
            Discount factor for future rewards
        epsilon: float
            Exploration probability in the epsilon-greedy policy.
        epsilon_min: float
            Minimum value of epsilon after decay. Prevents the agent from becoming fully greedy too early.
        epsilon_decay: float
            Multiplicative decay factor applied to epsilon after each episode. Controls how fast exploration decreases.
        n: int
            Number of steps used to compute the n-step return
        """
        super().__init__(env, alpha, gamma, epsilon, epsilon_min, epsilon_decay)
        self.n = n

    def train(self, episodes, num_points=5):
        """
        Train the agent using the n-step SARSA algorithm.

        Parameters
        episodes: int
            Number of training episodes
        num_points: int
            Number of intermediate nodes in the TSP environment

        Returns
        rewards_per_episode: list
            Total reward collected during each episode.
        """
        rewards_per_episode = []

        for episode in range(episodes):
            observation, _ = self.env.reset(num_points=num_points)
            state = self.get_state()
            valid_actions = self.get_valid_actions()

            if not valid_actions:
                rewards_per_episode.append(0)
                continue

            action = self.epsilon_greedy(state, valid_actions)

            # Store trajectory (states, actions, rewards)
            states = [state]
            actions = [action]
            rewards = [0.0]

            # T represents the time step when the episode terminates
            T = float("inf")
            # Current time step
            t = 0

            total_reward = 0.0

            while True:
                if t < T:
                    observation, reward, terminated, truncated, _ = self.env.step(
                        actions[t]
                    )

                    total_reward += reward
                    rewards.append(reward)

                    if terminated or truncated:
                        T = t + 1
                    else:
                        next_state = self.get_state()
                        next_valid_actions = self.get_valid_actions()
                        if not next_valid_actions:
                            T = t + 1
                        else:
                            next_action = self.epsilon_greedy(
                                next_state, next_valid_actions
                            )
                            states.append(next_state)
                            actions.append(next_action)
                # Time index for updating Q-values
                tau = t - self.n + 1

                if tau >= 0:
                    G = 0.0
                    for i in range(tau + 1, min(tau + self.n, T) + 1):
                        G += (self.gamma ** (i - tau - 1)) * rewards[i]

                    if tau + self.n < T:
                        G += (self.gamma**self.n) * self.get_q_value(
                            states[tau + self.n], actions[tau + self.n]
                        )

                    state_tau = states[tau]
                    action_tau = actions[tau]

                    # Clip return to avoid large updates and improve numerical stability
                    G = max(-1000.0, min(1000.0, G))

                    current_q = self.get_q_value(state_tau, action_tau)
                    new_q = current_q + self.alpha * (G - current_q)
                    self.update_q(state_tau, action_tau, new_q)

                if tau == T - 1:
                    break

                t += 1

            rewards_per_episode.append(total_reward)
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return rewards_per_episode
