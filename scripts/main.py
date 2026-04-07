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

    Parameters:
    agent : BaseAgent
        Trained RL agent
    state : tuple
        Current state
    valid_actions : list[int]
        List of valid actions

    Returns:
    action : int or None
        Selected action
    """
    if not valid_actions:
        return None
    
    best_actions = []
    best_value = float("-inf")

    for a in valid_actions:
        v = agent.get_combined_q(state, a)
        if v is None or np.isnan(v):
            continue
        if v > best_value:
            best_value = v
            best_actions = [a]
        elif v == best_value:
            best_actions.append(a)

    if not best_actions:
        return random.choice(valid_actions)
        
    return random.choice(best_actions)

def nearest_neighbor(env, num_points):
    """
    Greedy heuristic baseline (Nearest Neighbor).

    At each step, selects the closest unvisited node.
    Used for comparison with RL performance.

    Parameters:
    env : TSPEnvironment
        Environment instance
    num_points : int
        Number of intermediate nodes

    Returns:
    total_distance : float
        Total tour length
    path : list[int]
        Sequence of visited node indices
    """
    env.reset(num_points)

    terminated = False
    truncated = False

    while not (terminated or truncated):
        current = env.current_node

        # Build list of valid actions (unvisited nodes)
        valid_actions = []
        for i in range(len(env.nodes) - 1):
            if i not in env.visited:
                valid_actions.append(i - 1) # convert node index -> action index

        if not valid_actions:
            break

        best_action = None
        best_distance = float("inf")

        # Select nearest unvisited node
        for a in valid_actions:
            node_index = a + 1  # action -> node
            dist = env._euclidean_distance(current, node_index)

            if dist < best_distance:
                best_distance = dist
                best_action = a

        _, _, terminated, truncated, _ = env.step(best_action)

    return env.total_distance, env.path

def random_policy(env, num_points):
    """
    Random baseline policy.

    Selects actions uniformly at random from the set of valid actions.
    This serves as a lower-bound baseline for comparison.

    Parameters:
    env : TSPEnvironment
        Environment instance
    num_points : int
        Number of intermediate nodes

    Returns:
    total_distance : float
        Total tour length
    """
    env.reset(num_points)

    terminated = False
    truncated = False

    while not (terminated or truncated):
        # Build list of valid actions (unvisited nodes)
        valid_actions = []
        for i in range(1, len(env.nodes)-1):
            if i not in env.visited:
                valid_actions.append(i-1)

        if not valid_actions:
            break

        action = random.choice(valid_actions)
        _, _, terminated, truncated, _ = env.step(action)

    return env.total_distance

def evaluate_policy(env_class, agent, num_points, runs=5):
    """
    Evaluate trained RL agent against baseline heuristics.

    The function runs multiple independent episodes and compares:
    - RL agent (greedy policy, no exploration)
    - Nearest Neighbor heuristic
    - Random policy

    Parameters:
    env_class : class
        Environment class (TSPEnvironment)
    agent : BaseAgent
        Trained RL agent
    num_points : int
        Number of intermediate nodes
    runs : int
        Number of evaluation runs

    Returns:
    rl_mean : float
        Mean distance achieved by RL agent
    rl_std : float
        Standard deviation of RL performance
    nn_mean : float
        Mean distance of nearest neighbor heuristic
    rand_mean : float
        Mean distance of random policy
    """
    rl_distances = []
    nn_distances = []
    rand_distances = []
    original_env = agent.env

    for _ in range(runs):
        # RL
        env = env_class("config.json")
        env.reset(num_points)

        agent.env = env
        state = agent.get_state()
        terminated = False
        truncated = False

        steps = 0
        # Limit steps to avoid infinite loops
        max_steps = num_points + 5

        while not (terminated or truncated) and steps < max_steps:
            valid_actions = agent.get_valid_actions()
            steps += 1
            if not valid_actions:
                break

            action = greedy_action(agent, state, valid_actions)
            _, _, terminated, truncated, _ = env.step(action)
            state = agent.get_state()

        rl_distances.append(env.total_distance)

        # NN
        # Evaluate Nearest Neighbor baseline
        nn_env = env_class("config.json")
        nn_d, _ = nearest_neighbor(nn_env, num_points)
        nn_distances.append(nn_d)

        # RANDOM
        # Evaluate Random baseline
        rand_env = env_class("config.json")
        rand_d = random_policy(rand_env, num_points)
        rand_distances.append(rand_d)

        agent.env = original_env

    return (
        np.mean(rl_distances),
        np.std(rl_distances),
        np.mean(nn_distances),
        np.mean(rand_distances),
    )

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
            print("Visited:", len(env.visited), "/", len(env.nodes))

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
                print("Total distance:", env.total_distance)

                rl_mean, rl_std, nn_mean, rand_mean = evaluate_policy(TSPEnvironment, agent, num_points, runs=5)

                print(f"RL average distance: {rl_mean:.2f} ± {rl_std:.2f}")
                print(f"NN average distance: {nn_mean:.2f}")
                print(f"Random average distance: {rand_mean:.2f}")

                improvement = (nn_mean - rl_mean) / (nn_mean + 1e-8) * 100
                print(f"Improvement over NN: {improvement:.2f}%")
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