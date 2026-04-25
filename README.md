# Tabular Reinforcement Learning for Traveling Salesman Navigation

This project addresses a navigation variant of the Traveling Salesman Problem (TSP) using
tabular Reinforcement Learning methods.  
A mobile robot starts at position (0, 0), must visit a set of randomly generated intermediate
points, and finally reach the goal at (100, 100), while minimizing the total traveled distance.

The problem is formulated as a Markov Decision Process (MDP) and solved using classical
temporal-difference control algorithms.

---

## Problem Description

- Start position: (0, 0)  
- Goal position: (100, 100)  
- Intermediate points: N ∈ {5, 10, 15, 20}  
- Points are uniformly sampled with integer coordinates from: X ∈ [10, 90], Y ∈ [10, 90]
- Random seed: 42 (fixed for all experiments)


The robot must visit all intermediate points and reach the goal with minimal total path length. Observations consist of normalized Euclidean distances from the current position to each intermediate node, along with a binary visited mask. Actions correspond to selecting the next point to visit. The environment follows the Gymnasium-style interface (`reset`, `step`) and simulates an MDP. 

---

## Environment

The custom `TSPEnvironment` class implements the following interface:

| Method | Description |
|---|---|
| `reset(num_points)` | Starts a new episode; returns initial observation and info dict |
| `step(action)` | Executes action; returns `(observation, reward, terminated, truncated, info)` |
| `render()` | Prints current environment state to stdout |
| `close()` | Gymnasium API placeholder |

**Observation space:** `[distances_to_nodes, visited_mask]` — a vector of length `2 × N`, where distances are normalized to `[0, 1]` by the maximum possible workspace diagonal.

**Action space:** `Discrete(N)` — each action selects one of the N intermediate nodes.

The map (intermediate node positions) is generated once at initialization using the fixed seed and reused across all episodes, ensuring a stable training target.

**State representation** used by agents:

```python
state = (current_node, visited_mask, remaining, dist_to_goal)
```

---

## Reward Function

The reward is defined as the negative normalized Euclidean distance for each move.
```
r(s, a) = -(distance / max_dist)
```
Invalid actions (revisiting already visited nodes) receive a penalty of -5.
The episode terminates when the goal is reached.
When all N intermediate nodes are visited, the agent automatically moves to the goal and receives a completion bonus:
```
bonus = 5.0 / (1.0 + total_distance / (N × max_dist))
```

This bonus rewards shorter total routes and encourages the agent to complete the full tour rather than getting stuck in repeated invalid actions.

This formulation directly minimizes the total traveled distance. The cumulative reward corresponds to the negative total path length.

---

## Implemented Algorithms

The following tabular reinforcement learning algorithms are evaluated:

- SARSA (On-policy TD control)
- Q-learning (Off-policy TD control)
- Double Q-learning
- n-step SARSA (n ≥ 10)

All methods use discrete Q-tables (no neural networks) with ε-greedy action selection.

### SARSA (On-policy TD control)

Updates Q-values using the action actually taken in the next state:

```
Q(s, a) ← Q(s, a) + α [r + γ · Q(s', a') − Q(s, a)]
```

### Q-learning (Off-policy TD control)

Updates Q-values using the greedy maximum over the next state:

```
Q(s, a) ← Q(s, a) + α [r + γ · max_a' Q(s', a') − Q(s, a)]
```

### Double Q-learning

Maintains two independent Q-tables (Q1, Q2) to reduce overestimation bias. At each step, one table is randomly selected for the update while the other is used for evaluation:

```
Q1(s, a) ← Q1(s, a) + α [r + γ · Q2(s', argmax_a' Q1(s', a')) − Q1(s, a)]
```

Action selection uses the combined estimate `Q1(s, a) + Q2(s, a)`.

### n-step SARSA (n = 20)

Extends SARSA by accumulating rewards over n steps before updating:

```
G_t = r_{t+1} + γ·r_{t+2} + ... + γ^(n−1)·r_{t+n} + γ^n · Q(s_{t+n}, a_{t+n})
Q(s_t, a_t) ← Q(s_t, a_t) + α [G_t − Q(s_t, a_t)]
```


---

## Hyperparameters

| Parameter           | Value   |
|---------------------|---------|
| Learning_rate (α)   | 0.1     |
| Discount_factor (γ) | 0.99    |
| Epsilon (ε) start   | 1.0     |
| Epsilon min         | 0.05    |
| Epsilon decay       | 0.99995 |
| n (for n-step SARSA)| 20      |


---

## Installation

Install dependencies:

```bash
pip install -r requirements.txt
```
---

## Running Experiments

To run training and evaluation:

```bash 
python main.py 
```

Configuration is loaded from `config.json`. To change the number of episodes, learning rate, which algorithms to run, or the value of n, edit that file directly.

---

## Results and Visualization

The project generates and stores the following outputs:
- Reward curves (per episode and smoothed)
- Comparison plots across algorithms and problem sizes
- Final routes for each algorithm
- Animated visualizations of robot trajectories

All results are saved in the 'results/' directory.

---
