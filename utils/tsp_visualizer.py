import matplotlib.pyplot as plt

class TSPVisualizer:
    """
    Utility class used for visualizing TSP solutions.
    """
    def __init__(self, points):
        """
        Parameters 
        points: list of tuples
            Coordinates of cities[(x1, y1), (x2, y2), ...]
        """
        self.points = points

    def plot_route(self, route, title = "TSP Solution"):
        """
        Plot the TSP route.

        Parameters
        route: list[int]
            Order of visited cities
        title: str
        """
        x = [self.points[i][0] for i in route]
        y = [self.points[i][1] for i in route]

        plt.figure()

        plt.plot(x, y, marker = "o")

        for i, (px, py) in enumerate(self.points):
            plt.text(px, py, str(i))

        plt.title(title)
        plt.xlabel("X")
        plt.ylabel("Y")

        plt.show()