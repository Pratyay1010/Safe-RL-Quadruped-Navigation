# Safe RL for Quadruped Navigation

<p align="center">
  <img src="assets/doggo_policy.gif" width="750"/>
</p>

<p align="center">
  PPO-Lagrangian for safe quadruped navigation in Safety-Gymnasium.
</p>

---

## Overview

This project explores **Safe Reinforcement Learning** for continuous robotic navigation using a PPO-Lagrangian agent trained in the `SafetyDoggoGoal2-v0` environment from Safety-Gymnasium.

The implementation focuses on:

- constraint-aware policy optimization
- safe exploration under cost constraints
- continuous robotic control
- compact research-oriented RL implementation

The repository is intentionally lightweight and designed to resemble a clean experimental robotics research codebase.

---

## Motivation

Traditional reinforcement learning agents can achieve high reward while still exhibiting unsafe behavior.

In robotics and autonomous systems, maximizing reward alone is often insufficient — agents must also satisfy safety constraints during training and deployment.

This project investigates:

- constrained policy optimization
- reward-cost tradeoffs
- safe navigation behavior
- PPO-Lagrangian for CMDPs (Constrained Markov Decision Processes)

using the Doggo robot environment from Safety-Gymnasium.

---

## Features

- PPO-Lagrangian implementation from scratch
- Separate reward and cost critics
- Continuous control with MuJoCo
- Constraint-aware policy optimization
- Compact modular codebase
- Evaluation pipeline with CSV logging

---

## Environment

- Environment: `SafetyDoggoGoal2-v0`
- Framework: Safety-Gymnasium
- Physics Backend: MuJoCo
- Deep Learning Framework: PyTorch

---

## Repository Structure

```text
Safe-RL-Quadruped-Navigation/
│
├── assets/
│   └── doggo_policy.gif
│
├── checkpoints/
│
├── configs/
│   └── default.py
│
├── results/
│   └── evaluation.csv
│
├── src/
│   ├── agent.py
│   ├── buffer.py
│   ├── networks.py
│   ├── train.py
│   └── evaluate.py
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/Safe-RL-Quadruped-Navigation.git

cd Safe-RL-Quadruped-Navigation
```

### 2. Create environment

```bash
conda create -n safe-rl python=3.10

conda activate safe-rl
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install Safety-Gymnasium

```bash
pip install safety-gymnasium
```

---

## Training

Run PPO-Lagrangian training:

```bash
python src/train.py
```

The best checkpoint will be saved to:

```text
checkpoints/best_model.pth
```

---

## Evaluation

Run evaluation using the saved checkpoint:

```bash
python src/evaluate.py
```

Evaluation statistics are saved to:

```text
results/evaluation.csv
```

---

## About `evaluation.csv`

The evaluation file stores per-episode performance metrics collected during policy evaluation.

| Column | Description |
|---|---|
| `episode` | Evaluation episode index |
| `return` | Total episodic reward |
| `cost` | Total episodic safety cost |

This enables quick analysis of:

- policy reward performance
- safety constraint violations
- reward-cost tradeoffs across episodes

---

## PPO-Lagrangian

The agent optimizes policy reward while enforcing safety constraints using a Lagrangian penalty formulation.

The implementation includes:

- stochastic Gaussian policy
- clipped PPO objective
- separate reward and cost critics
- adaptive safety penalty coefficient
- Generalized Advantage Estimation (GAE)

The approach follows constrained reinforcement learning formulations commonly used in safe robotic learning.

---

## Results

| Metric | Value |
|---|---|
| Mean Return | -- |
| Mean Cost | -- |

---

## References

### PPO
- Schulman et al. — *Proximal Policy Optimization Algorithms*  
  https://arxiv.org/abs/1707.06347

### Safe RL / PPO-Lagrangian
- Ray et al. — *Benchmarking Safe Exploration in Deep Reinforcement Learning*  
  https://arxiv.org/abs/1910.01708

### Safe RL
- Ray et al. — *Benchmarking Safe Exploration in Deep Reinforcement Learning*  
  https://arxiv.org/abs/1910.01708

### Safety-Gymnasium
- Ji et al. — *Safety-Gymnasium: Safe Reinforcement Learning Benchmarking Platform*  
  https://arxiv.org/abs/2310.20431