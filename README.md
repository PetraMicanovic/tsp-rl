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
- Intermediate points: N = {5, 10, 15, 20}  
- Points are uniformly sampled with integer coordinates from: X ∈ [10, 90], Y ∈ [10, 90]


The robot must visit points sequentially and reach the goal with minimal total path length. Observations consist of Euclidean distances to remaining points. Actions correspond to selecting the next point to visit. The environment follows the Gymnasium-style interface (`reset`, `step`) and simulates an MDP. A fixed random seed is used for all experiments to ensure reproducibility. 

To reduce the size of the state space, distances are discretized by rounding to the nearest multiple of 5.

---

## Implemented Algorithms

The following tabular reinforcement learning algorithms are evaluated:

- SARSA (On-policy TD control)
- Q-learning (Off-policy TD control)
- Double Q-learning
- n-step SARSA (n ≥ 10)

All methods use discrete state-action tables (no neural networks).

---

## Reward Function

The reward is defined as the negative Euclidean distance for each move.
Invalid actions (revisiting already visited nodes) receive a penalty of -200.
The episode terminates when the goal is reached.

This formulation directly minimizes the total traveled distance. The cumulative reward corresponds to the negative total path length.

---

## Hyperparameters

Key hyperparameters include:
- learning_rate (α)
- discount_factor (γ)
- epsilon (ε-greedy exploration)
- epsilon_decay
- n (for n-step SARSA)

Their impact on performance is analyzed experimentally.

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
---

## Results and Visualization

The project generates and stores the following outputs:
- Reward curves (per episode and smoothed)
- Comparison plots across algorithms and problem sizes
- Final routes for each algorithm
- Animated visualizations of robot trajectories

All results are saved in the 'results/' directory.

---


