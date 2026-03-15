from base_agent import BaseAgent

class NStepSARSAAgent(BaseAgent):
    def __init__(self, env, alpha, gamma, epsilon, n = 10):
        super().__init__(env, alpha, gamma, epsilon)
        self.n  = n
    
    def train(self, episodes, num_points = 5):
        rewards_per_episode = []

        for episode in range(episodes):
            observation, _ = self.env.reset(num_points = num_points)
            state = self.get_state()
            valid_actions = self.get_valid_actions()
            action = self.epsilon_greedy(state, valid_actions)

            states =[state]
            actions = [action]
            rewards = [0]

            T = float("inf")
            t = 0

            total_reward = 0.0

            while True:
                if t < T:
                    observation, reward, terminated, truncated, _ = self.env.step(actions[t])

                    total_reward += reward
                    rewards.append(reward)

                    if terminated or truncated:
                        T = t + 1
                    else:
                        next_state = self.get_state()
                        next_valid_actions = self.get_valid_actions()
                        next_action = self.epsilon_greedy(next_state, next_valid_actions)

                        states.append(next_state)
                        actions.append(next_action)

                tau = t - self.n + 1

                if tau >= 0:
                    G = 0.0
                    for i in range(tau + 1, min(tau + self.n, T) + 1):
                        G += (self.gamma ** (i - tau - 1)) * rewards[i]

                    if tau + self.n < T:
                        G += (self.gamma ** self.n) * self.get_q_value(states[tau + self.n], actions[tau + self.n]) 

                    state_tau = states[tau]
                    action_tau = actions[tau]

                    current_q = self.get_q_value(state_tau, action_tau)
                    new_q = current_q + self.alpha * (G -current_q)
                    self.update_q(state_tau, action_tau, new_q)

                if tau == T - 1:
                    break

                t += 1
            
            rewards_per_episode.append(total_reward)
        return rewards_per_episode