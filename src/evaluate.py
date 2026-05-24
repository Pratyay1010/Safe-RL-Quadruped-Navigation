import safety_gymnasium
import torch
import numpy as np
import pandas as pd
import os

from configs.default import CONFIG
from agent import Agent


MODEL_PATH = "checkpoints/best_model.pth"
RESULTS_PATH = "results/evaluation.csv"
NUM_EPISODES = 5

os.makedirs("results", exist_ok=True)

print(f"Loading model from {MODEL_PATH}")
env = safety_gymnasium.make(CONFIG["ENV_ID"])
agent = Agent(env, CONFIG)

if not agent.load_model(MODEL_PATH):
    exit(1)

print(f"Evaluating over {NUM_EPISODES} episodes...")
all_rewards, all_costs = [], []

for episode in range(NUM_EPISODES):
    obs, info = env.reset()
    done = False
    ep_reward, ep_cost = 0.0, 0.0

    while not done:
        action = agent.get_action(obs)
        obs, reward, cost, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        ep_reward += reward
        ep_cost += cost

    all_rewards.append(ep_reward)
    all_costs.append(ep_cost)
    print(f"Episode {episode + 1}/{NUM_EPISODES}: reward={ep_reward:.2f}, cost={ep_cost:.2f}")

df = pd.DataFrame({
    'episode': range(1, NUM_EPISODES + 1),
    'return': all_rewards,
    'cost': all_costs
})
df.to_csv(RESULTS_PATH, index=False)

print(f"\nResults saved to {RESULTS_PATH}")
print(f"Mean return: {np.mean(all_rewards):.2f} ± {np.std(all_rewards):.2f}")
print(f"Mean cost: {np.mean(all_costs):.2f} ± {np.std(all_costs):.2f}")