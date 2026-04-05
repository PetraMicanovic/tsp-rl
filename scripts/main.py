import os
import sys
import json
import numpy as np
import random
import copy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from environment.env import TSPEnvironment
from agents.sarsa_agent import SARSAAgent
from agents.q_learning_agent import QLearningAgent
from agents.double_q_learning import DoubleQLearningAgent
from agents.n_step_sarsa import NStepSARSAAgent

from utils.training_plotter import TrainingPlotter
from utils.tsp_visualizer import TSPVisualizer

def load_config(path = "config.json"):
    """
    Load configuration file.

    Parameters
    path: str
        Path to JSON configuration file.
    Returns
    dict
        Parsed configuration dictionary.
    """
    with open(path, 'r') as f:
        return json.load(f)
    
def create_agent(name, env, config):
    """
    Function for creating RL agents.

    Parameters:
    name: str
        Name of the algorithm ("sarsa", "q_learning", etc.)
    env: TSPEnvironment
        Environment instance
    config: dict
        Configuration dictionary
    Returns:
    BaseAgent
        Initialized agent instance
    """
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
    This avoids randomness from epsilon-greedy policy.

    Parameters
    agent: BaseAgent
    state: tuple
    valid_actions: list[int]
    Returns
    best_action: int
        Action with highest Q-value
    """
    best_action = valid_actions[0]
    best_value = agent.get_combined_q(state, best_action)

    for a in valid_actions[1:]:
        v = agent.get_combined_q(state, a)
        if v > best_value:
            best_value = v
            best_action = a
        
    return best_action

def run_experiment(agent_name, param_name, values, config, episodes, num_points):
    """
    Runs a hyperparameter experiment for a given algorithm.

    Parameters:
    agent_name: str
        Algorithm name
    param_name: str
        Name of the hyperparameter to vary
    values: list
        List of values for the hyperparameter
    config: dict
        Base configuration 
    episodes: int
        Number of training episodes
    num_points: int
        Number of TSP nodes

    Returns:
    results: dict
        Dictionary mapping parameter values to reward curves
    """
    results = {}

    for v in values:
        print(f"{agent_name} with {param_name} = {v}")

        local_config = copy.deepcopy(config)
        env = TSPEnvironment("config.json")
        env.reset(num_points)

        if param_name == "epsilon_decay":
            local_config["training"]["epsilon_decay"] = v
        elif param_name == "learning_rate":
            local_config["training"]["learning_rate"] = v
        elif param_name == "n":
            local_config["algorithms"]["n_step_sarsa"]["n"] = v

        agent = create_agent(agent_name, env, local_config)
        rewards = agent.train(episodes, num_points=num_points)

        results[f"{param_name}={v}"] = rewards

    return results

def main():
    """
    Main experiment pipeline.
    
    Workflow:
    1. Load configuration
    2. Initialize environment
    3. Train agents for different number of points
    4. Save reward plots
    5. Evaluate final learned policy
    6. Visualize routes
    7. Plot comparison of diffrent hyperparameters for a Double Q-learning and n-step SARSA algorithm.

    """
    config = load_config()
    random.seed(config["environment"]["random_seed"])
    np.random.seed(config["environment"]["random_seed"])

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

            env = TSPEnvironment("config.json")
            env.reset(num_points)
            
            agent = create_agent(algorithm_name, env, config)
            rewards = agent.train(episodes, num_points = num_points)

            all_results[num_points][algorithm_name] = rewards
            # Debug info
            print("Visited:", len(env.visited), "/", len(env.nodes)-1)

            # Save reward curve
            if config["evaluation"]["save_reward_curves"]:
                plotter.plot_rewards(rewards, algorithm_name, num_points)

            # Evaluate
            if config["evaluation"]["save_final_paths"]:
                env.reset(num_points)
                state = agent.get_state()

                terminated = False
                truncated = False

                while not (terminated or truncated):
                    valid_actions = agent.get_valid_actions()
                    # if no valid actions -> force termination
                    if not valid_actions:
                        terminated = True
                        continue 

                    action = greedy_action(agent, state, valid_actions)

                    obs, reward, terminated, truncated, info = env.step(action)
                    state = agent.get_state()

                # Debug final route
                print("Final route:", env.path)

                # Visualization
                if len(env.path) > 1:
                    visualizer = TSPVisualizer(env.nodes)
                    visualizer.plot_route(env.path, algorithm_name, num_points)
                    visualizer.animate_route(env.path, algorithm_name, num_points)

        if config["evaluation"]["compare_algorithms"]:
            plotter.compare_algorithms(all_results[num_points],num_points)
    plotter.compare_N_for_each_algorithm(all_results)

    # Hyperparameter analysis (N = 20)
    print("\n Running hyperparameter analysis (N = 20)")

    # Double Q-learning
    ## Epsilon decay
    decay_values = [0.9999, 0.9997, 0.9995]
    dq_ed_results = run_experiment("double_q_learning", "epsilon_decay", decay_values, config, episodes, 20)
    plotter.compare_algorithms(dq_ed_results, "DoubleQ_epsilon_decay_20")

    ## Learning rate
    learning_rate_values = [0.2, 0.1, 0.05]
    dq_lr_results = run_experiment("double_q_learning", "learning_rate", learning_rate_values, config, episodes, 20)
    plotter.compare_algorithms(dq_lr_results, "DoubleQ_learning_rate_20")

    # n-step SARSA
    ## Epsilon decay
    nSARSA_ed_results = run_experiment("n_step_sarsa", "epsilon_decay", decay_values, config, episodes, 20)
    plotter.compare_algorithms(nSARSA_ed_results, "n_step_SARSA_epsilon_decay_20")

    ## Learning rate
    nSARSA_lr_results = run_experiment("n_step_sarsa", "learning_rate", learning_rate_values, config, episodes, 20)
    plotter.compare_algorithms(nSARSA_lr_results, "n_step_SARSA_learning_rate_20")

    ## N
    n_values = [10, 15, 20]
    nSARSA_n_results = run_experiment("n_step_sarsa", "n", n_values, config, episodes, 20)
    plotter.compare_algorithms(nSARSA_n_results, "n_step_SARSA_n_20")

    print("\n----------- Training complete -----------------")


if  __name__ == "__main__":
    main()