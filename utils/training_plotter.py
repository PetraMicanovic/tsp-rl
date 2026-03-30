import matplotlib.pyplot as plt 
import os
import numpy as np

class TrainingPlotter:
    """
    The Training plotter class provides methods for:
        - plotting reward per episode
        - plotting moving average reward
        - comparing multiple algorithms on a single graph
    """
    def __init__(self, reward_dir = "results/reward_curves", comparison_dir = "results/comparisons", window = 1000):
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
        Compute moving average of rewards using convolution.

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
        
        rewards = np.array(rewards)
        kernel = np.ones(self.window) / self.window
        return np.convolve(rewards, kernel, mode = 'valid')
    
    def downsample(self, data, factor = 50):
        """
        Reduce the number of data points for plotting by selecting every k-th element.

        Parameters
        data: np.array
            Sequence of values (e.g. rewards or smoothed rewards) to be downsampled.
        factor: int
            Step size for sampling
        Returns:
        np.array
            Downsampled sequence with reduced number of points.
        """
        return np.array(data)[::factor]
    
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
        save_dir = os.path.join(self.reward_dir, algorithm_name)
        os.makedirs(save_dir, exist_ok = True)

        plt.style.use("seaborn-v0_8-white")     
        all_smoothed = []
   
        #Smoothed curve
        smoothed = self.moving_average(rewards)
        if len(smoothed) > 50:
            smoothed = self.downsample(smoothed, factor = 50)
        plt.plot(smoothed, linewidth = 2)
        all_smoothed.extend(smoothed)
        
        plt.xlabel("Episode (scaled)")
        plt.ylabel("Reward")
        plt.title(f"{algorithm_name} ({num_points} points)")
        all_smoothed = np.array(all_smoothed)
        y_min = np.min(all_smoothed)
        y_max = np.max(all_smoothed)
        margin = 0.05 * (y_max - y_min)
        plt.ylim(y_min - margin, y_max + margin)        
        filename = f"{algorithm_name}_{num_points}.png"
        save_path = os.path.join(save_dir, filename)

        plt.savefig(save_path, dpi = 150)
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
        plt.style.use("seaborn-v0_8-white")   
        all_smoothed = []

        for algorithm_name, rewards in results.items():
            smoothed = self.moving_average(rewards)
            if len(smoothed) > 50:
                smoothed = self.downsample(smoothed, factor = 50)
            plt.plot(smoothed, label = algorithm_name, linewidth = 2)
            all_smoothed.extend(smoothed)

        plt.xlabel("Episode (scaled)")
        plt.ylabel("Moving Average Reward")
        plt.title(f"Algorithm Comparison ({num_points} points)")
        plt.legend()
        all_smoothed = np.array(all_smoothed)
        y_min = np.min(all_smoothed)
        y_max = np.max(all_smoothed)
        margin = 0.05 * (y_max - y_min)
        plt.ylim(y_min - margin, y_max + margin)
        filename = f"comparison_{num_points}.png"
        save_path = os.path.join(self.comparison_dir, filename)

        plt.savefig(save_path, dpi = 150)
        plt.close()

    def compare_N_for_each_algorithm(self, all_results):
        """
        Plot comparison of algorithm for multiple points(5, 10, 15, 20) on a single graph.

        Parameters
        results: dict
            Dictionary mapping algorithm name to reward list.
        """
        plt.style.use("seaborn-v0_8-white")
        first_key = list(all_results.keys())[0]
        algorithms = all_results[first_key].keys()

        for algorithm in algorithms:
            plt.figure()
            all_smoothed = []

            for num_points, results in all_results.items():
                rewards = results[algorithm]
                smoothed = self.moving_average(rewards)
                if len(smoothed) > 50:
                    smoothed = self.downsample(smoothed, factor = 50)
                plt.plot(smoothed, label=f"N = {num_points}", linewidth = 2)
                all_smoothed.extend(smoothed)

            plt.xlabel("Episode (scaled)")
            plt.ylabel("Moving Average Reward")
            plt.title(f"{algorithm} - comparison across N")
            plt.legend()
            all_smoothed = np.array(all_smoothed)
            y_min = np.min(all_smoothed)
            y_max = np.max(all_smoothed)
            margin = 0.05 * (y_max - y_min)
            plt.ylim(y_min - margin, y_max + margin)
            filename = f"{algorithm}_all_N.png"
            save_path = os.path.join(self.comparison_dir, filename)

            plt.savefig(save_path, dpi = 150)
            plt.close()