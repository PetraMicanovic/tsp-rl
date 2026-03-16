import matplotlib.pyplot as plt 

class TrainingPlotter:
    """
    The Training plotter class provides methods for:
        - plotting reward per episode
        - plotting moving average reward
        - comparing multiple algorithms on a single graph
    """
    def __init__(self, window = 100):
        """
        Initialize the plotter.

        Parameters
        window: int
            Window size used for computing moving average smoothing.
        """
        self.window =  window
    
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
    
    def plot_rewards(self, rewards, title = "Training rewards"):
        """
        Plot reward per episode together with the moving average curve.

        Parameters
        rewards: list[float]
            Reward obtained in each episode
        title: str
            Title of the plot
        """
        plt.figure()
        plt.plot(rewards, label = "Reward per episode")
        
        #Smoothed curve
        smoothed = self.moving_average(rewards)
        plt.plot(range(len(smoothed)), smoothed, label = "Moving average")
        
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title(title)
        plt.legend()
        plt.show()

    def compare_algorithms(self, results, title = "Algorithm Comparison"):
        """
        Plot comparison of multiple algorithms on a single graph.

        Parameters
        results: dict
            Dictionary mapping algorithm name to reward list.
        title: str
            Title of the plot
        """
        plt.figure()

        for algorithm_name, rewards in results.items():
            smoothed = self.moving_average(rewards)
            plt.plot(smoothed, label = algorithm_name)
        plt.xlabel("Episode")
        plt.ylabel("Moving Average Reward")
        plt.title(title)
        plt.legend()
        plt.show()
