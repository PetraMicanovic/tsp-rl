import os
import sys
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from environment.env import TSPEnvironment
from agents.sarsa_agent import SARSAAgent
from agents.q_learning_agent import QLearningAgent
from agents.double_q_learning import DoubleQLearningAgent
from agents.n_step_sarsa import NStepSARSAAgent

from utils.training_plotter import TrainingPlotter
from utils.tsp_visualizer import TSPVisualizer

import numpy as np

def load_config(path = "config.json"):
    with open(path, 'r') as f:
        return json.load(f)
    
def create_agent(name, env, config):
    alpha = config["training"]["learning_rate"]
    gamma = config["training"]["discount_factor"]
    epsilon = config["training"]["epsilon_start"]
    epsilon_min = config["training"]["epsilon_min"]
    epsilon_decay = config["training"]["epsilon_decay"]

    if name == "sarsa":
        return SARSAAgent(env, alpha, gamma, epsilon, epsilon_min, epsilon_decay)
    elif name == "q_learning":
        return QLearningAgent(env, alpha, gamma, epsilon, epsilon_min, epsilon_decay)
    elif name == "double_q_learning":
        return DoubleQLearningAgent(env, alpha, gamma, epsilon, epsilon_min, epsilon_decay)
    elif name == "n_step_sarsa":
        n = config["algorithms"]["n_step_sarsa"]["n"]
        return NStepSARSAAgent(env, alpha, gamma, epsilon, epsilon_min, epsilon_decay, n)
    else:
        raise ValueError(f"Unknown algorithm: {name}")
    
def greedy_action(agent, state, valid_actions):
    """
    Select the best action deterministically.
    This function is used only during the evaluation, after training.
    This avoidness randomness from epsilon-greedy policy.
    """
    best_action = valid_actions[0]
    best_value = agent.get_combined_q(state, best_action)

    for a in valid_actions[1:]:
        v = agent.get_combined_q(state, a)
        if v > best_value:
            best_value = v
            best_action = a
        
    return best_action

def main():
    config = load_config()

    env = TSPEnvironment("config.json")

    plotter = TrainingPlotter()

    num_points_list = config["environment"]["num_intermediate_points"]
    episodes = config["training"]["episodes"]
    algorithms_config = config["algorithms"]

    # Directory to store results for comparison plots
    all_results = {}

    for num_points in num_points_list:
        print(f"\n Running experiments for {num_points} points")
        all_results[num_points] = {}

        for algorithm_name, enabled in algorithms_config.items():
            # skip disabled algorithms
            if isinstance(enabled, dict):
                if not enabled.get("enabled", False):
                    continue
            elif not enabled:
                continue
            print(f"Training: {algorithm_name}")

            agent = create_agent(algorithm_name, env, config)
            rewards = agent.train(episodes, num_points = num_points)

            all_results[num_points][algorithm_name] = rewards
            print("Visited:", len(env.visited), "/", len(env.nodes)-1)
            print("Path:", env.path)

            # Save reward curve
            if config["evaluation"]["save_reward_curves"]:
                plotter.plot_rewards(rewards, algorithm_name, num_points)

            # Save final route
            if config["evaluation"]["save_final_paths"]:
                env.reset(num_points)
                state = agent.get_state()
                route = [env.current_node]

                terminated = False
                truncated = False

                while not (terminated or truncated):
                    valid_actions = agent.get_valid_actions()
                    if not valid_actions:
                        terminated = True
                        continue 

                    action = greedy_action(agent, state, valid_actions)

                    obs, reward, terminated, truncated, info = env.step(action)
                    state = agent.get_state()
                    route.append(env.current_node)

                print("ENV PATH:", env.path)

                if len(env.path) > 1:
                    visualizer = TSPVisualizer(env.nodes)
                    visualizer.plot_route(env.path, algorithm_name, num_points)
                    visualizer.animate_route(env.path, algorithm_name, num_points)

        if config["evaluation"]["compare_algorithms"]:
            plotter.compare_algorithms(all_results[num_points],num_points)
            plotter.compare_N_for_each_algorithm(all_results)
    print("\n----------- Training complete -----------------")


if  __name__ == "__main__":
    main()