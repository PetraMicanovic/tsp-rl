import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from environment.env import TSPEnvironment
import numpy as np

env = TSPEnvironment()

obs, info = env.reset(num_points=5)

done = False

while not done:
    action = np.random.randint(env.action_space.n)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    env.render()

print("Total distance:", env.total_distance)


env.close()