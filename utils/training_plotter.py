import matplotlib.pyplot as plt 
import os

class TrainingPlotter:
    """
    The Training plotter class provides methods for:
        - plotting reward per episode
        - plotting moving average reward
        - comparing multiple algorithms on a single graph
    """
    def __init__(self, reward_dir = "results/reward_curves", comparison_dir = "results/comparisons", window = 100):
        """
        Initialize the plotter.

        Parameters
        reward_dir: str
            Directory where individual reward curves will be saved
        comparison_dir: string
            Directory where algorithm comparison graphs will be saved
        window: int
            Window size used for computing moving average smoothing.
        """
        self.reward_dir = reward_dir
        self.comparison_dir = comparison_dir
        self.window =  window

        #Create directories if they do not exist
        os.makedirs(self.reward_dir, exist_ok = True)
        os.makedirs(self.comparison_dir, exist_ok = True)
    
    def moving_average(self, rewards):
        """
        Compute moving average of rewards.

        Moving average is used to smooth noisy reinforcement learning reward curves so that learning trends become easier to observe.

        Parameters
        rewards: list[float]
            Reward obtained in each episode.
        
        Returns
        averages: list[float]
            Smoothed reward values
        """
        if len(rewards) < self.window:
            return rewards
        
        averages = []

        for i in range(len(rewards) - self.window + 1):
            window_avg = sum(rewards[i:i + self.window]) / self.window
            averages.append(window_avg)
        return averages
    
    def plot_rewards(self, rewards, algorithm_name, num_points):
        """
        Plot reward per episode together with the moving average curve.

        Parameters
        rewards: list[float]
            Reward obtained in each episode
        algorithm_name: str
            Name of the RL algorithm
        num_points: int
            Number of cities used in the environment
        """
        # Create algorithm directory 
        save_dir = os.path.join(save_dir, algorithm_name)
        os.makedirs(save_dir, exist_ok = True)

        plt.figure()
        plt.plot(rewards, label = "Reward per episode")
        
        #Smoothed curve
        smoothed = self.moving_average(rewards)
        plt.plot(range(len(smoothed)), smoothed, label = "Moving average")
        
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title(f"{algorithm_name} ({num_points} points)")
        plt.legend()

        filename = f"{algorithm_name}_{num_points}.png"
        save_path = os.path.join(self.reward_dir, filename)

        plt.savefig(save_path)
        plt.close()

    def compare_algorithms(self, results, num_points):
        """
        Plot comparison of multiple algorithms on a single graph.

        Parameters
        results: dict
            Dictionary mapping algorithm name to reward list.
        num_points: int
            Number of points in the TSP problem
        """
        plt.figure()

        for algorithm_name, rewards in results.items():
            smoothed = self.moving_average(rewards)
            plt.plot(smoothed, label = algorithm_name)
        plt.xlabel("Episode")
        plt.ylabel("Moving Average Reward")
        plt.title(f"Algorithm Comparison ({num_points} points)")
        plt.legend()
        
        filename = f"comparison_{num_points}.png"
        save_path = os.path.join(self.comparison_dir, filename)

        plt.savefig(save_path)
        plt.close()

    def compare_N_for_each_algorithm(self, all_results):
        """
        Plot comparison of algorithm for multiple points(5, 10, 15, 20) on a single graph.

        Parameters
        results: dict
            Dictionary mapping algorithm name to reward list.
        """

        first_key = list(all_results.keys())[0]
        algorithms = all_results[first_key].keys()

        for algorithm in algorithms:
            plt.figure()

            for num_points, results in all_results.items():
                rewards = results[algorithm]
                smoothed = self.moving_average(rewards)

                plt.plot(smoothed, label=f"N = {num_points}")

            plt.xlabel("Episode")
            plt.ylabel("Moving Average Reward")
            plt.title(f"{algorithm} - comparison across N")
            plt.legend()
            filename = f"{algorithm}_all_N.png"
            save_path = os.path.join(self.comparison_dir, filename)

            plt.savefig(save_path)
            plt.close()