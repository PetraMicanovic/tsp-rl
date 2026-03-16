import matplotlib.pyplot as plt
import os

class TSPVisualizer:
    """
    Utility class used for visualizing TSP solutions.
    """
    def __init__(self, points, base_dir = "results/routes"):
        """
        Parameters 
        points: list of tuples
            Coordinates of cities[(x1, y1), (x2, y2), ...]
        base_dir: str
            Base directory for saving route visualizations.
        """
        self.points = points
        self.base_dir = base_dir

    def plot_route(self, route, algorithm_name, num_points):
        """
        Plot the TSP route.

        Parameters
        route: list[int]
            Order of visited cities
        algorithm_name: str
            Name of the RL algorithm
        num_points: int
            Number of cities used in the environment
        """
        # Create algorithm directory 
        save_dir = os.path.join(self.base_dir, algorithm_name)
        os.makedirs(save_dir, exist_ok = True)

        x = [self.points[i][0] for i in route]
        y = [self.points[i][1] for i in route]

        plt.figure()

        plt.plot(x, y, marker = "o")

        for i, (px, py) in enumerate(self.points):
            plt.text(px, py, str(i))

        plt.title(f"{algorithm_name} - TSP Route ({num_points} points)")
        plt.xlabel("X")
        plt.ylabel("Y")
        
        filename = f"route_{num_points}.png"
        save_path = os.path.join(save_dir, filename)

        plt.savefig(save_path)
        plt.close()