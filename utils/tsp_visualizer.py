import matplotlib.pyplot as plt

class TSPVisualizer:
    def __init__(self, points):
        self.points = points

    def plot_route(self, route, title = "TSP Solution")
        
        x = [self.points[i][0] for i in route]
        y = [self.points[i][1] for i in route]

        plt.figure()

        plt.plot(x,y, marker = "o")

        for i, (px, py) in enumerate(self.points):
            plt.text(px, py, str(i))

        plt.title(title)
        plt.xlabel("X")
        plt.ylabel("Y")

        plt.show()