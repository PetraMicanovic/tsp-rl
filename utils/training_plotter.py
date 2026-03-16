import matplotlib.pyplot as plt 

class TrainingPlotter:
    def __init__(self, window = 100):
        self.window =  window
    
    def moving_average(self, rewards):
        if len(rewards) < self.window:
            return rewards
        
        averages = []

        for i in range(len(rewards) - self.window + 1):
            window_avg = sum(rewards[i:i + self.window]) / self.window
            averages.append(window_avg)
        return averages
    
    def plot_rewards(self, rewards, title = "Training rewards"):
        plt.figure()
        plt.plot(rewards, label = "Reward per episode")
        
        smoothed = self.moving_average(rewards)
        plt.plot(range(len(smoothed)), smoothed, label = "Moving average")
        
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title(title)
        plt.legend()
        plt.show()

    def compare_algorithms(self, results, title = "Algorithm Comparison"):
        plt.figure()

        for algorithm_name, rewards in results.items():
            smoothed = self.moving_average(rewards)
            plt.plot(smoothed, label = algorithm_name)
        plt.xlabel("Episode")
        plt.ylabel("Moving Average Reward")
        plt.title(title)
        plt.legend()
        plt.show()
