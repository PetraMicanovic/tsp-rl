import matplotlib.pyplot as plt
import os
import matplotlib.animation as animation


class TSPVisualizer:
    """
    Utility class used for visualizing TSP solutions.
    """

    def __init__(self, points, base_dir="results/routes"):
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
        os.makedirs(save_dir, exist_ok=True)

        x = []
        y = []
        for i in route:
            x.append(self.points[i][0])
            y.append(self.points[i][1])

        fig, ax = plt.subplots()

        ax.plot(x, y, color="blue", linewidth=1, marker="o")

        # Draw arrows
        for i in range(len(route) - 1):
            start = self.points[route[i]]
            end = self.points[route[i + 1]]

            dx = end[0] - start[0]
            dy = end[1] - start[1]

            ax.arrow(
                start[0],
                start[1],
                dx,
                dy,
                head_width=1.5,
                length_includes_head=True,
                color="red",
                alpha=1,
            )

        for i, (px, py) in enumerate(self.points):
            plt.text(px, py, str(i))

        plt.title(f"{algorithm_name} - TSP Route ({num_points} points)")
        plt.xlabel("X")
        plt.ylabel("Y")

        filename = f"route_{num_points}.png"
        save_path = os.path.join(save_dir, filename)

        plt.savefig(save_path)
        plt.close()

    def animate_route(self, route, algorithm_name, num_points):
        """
        Create and save an animation of the TSP route being constructed.

        Parameters
        route: list[int]
            Order in which cities are visited
        algorithm_name: str
            Name of the RL algorithm
        num_points: int
            Number of cities in the environment
        """

        save_dir = os.path.join("results/animations", algorithm_name)
        os.makedirs(save_dir, exist_ok=True)

        fig, ax = plt.subplots()

        x_points = []
        y_points = []
        for p in self.points:
            x_points.append(p[0])
            y_points.append(p[1])

        ax.scatter(x_points, y_points)

        ax.set_xlim(min(x_points) - 5, max(x_points) + 5)
        ax.set_ylim(min(y_points) - 5, max(y_points) + 5)

        for i, (px, py) in enumerate(self.points):
            ax.text(px, py, str(i))

        (line,) = ax.plot([], [], marker="o")

        ax.set_title(f"{algorithm_name} - TSP Animation ({num_points} points)")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

        def update(frame):
            partial_route = route[: frame + 1]

            x = []
            y = []
            for i in partial_route:
                x.append(self.points[i][0])
                y.append(self.points[i][1])

            line.set_data(x, y)

            for p in list(ax.patches):
                p.remove()

            # Draw arrows showing direction of travel
            for i in range(len(partial_route) - 1):
                start = self.points[partial_route[i]]
                end = self.points[partial_route[i + 1]]

                dx = end[0] - start[0]
                dy = end[1] - start[1]

                ax.arrow(
                    start[0],
                    start[1],
                    dx,
                    dy,
                    head_width=1.5,
                    length_includes_head=True,
                    color="red",
                    alpha=0.7,
                )
            return (line,)

        ani = animation.FuncAnimation(
            fig, update, frames=len(route), interval=500, blit=False
        )

        filename = f"tsp_animation_{num_points}.gif"
        save_path = os.path.join(save_dir, filename)

        ani.save(save_path, writer="pillow")

        plt.close()
