#!/usr/bin/env python3

import torch
import safety_gymnasium

from configs.default import CONFIG
from agent import Agent


def main():
    cfg = CONFIG.copy()

    env = safety_gymnasium.make(
        cfg["ENV_ID"],
        render_mode="rgb_array"
    )

    agent = Agent(env, cfg)

    print(f"Device: {cfg['DEVICE']}")

    try:
        agent.train()

    except KeyboardInterrupt:
        print("\nInterrupted — saving checkpoint...")

        if agent.best_model:
            torch.save(agent.best_model, cfg["SAVE_PATH"])

        env.close()


if __name__ == "__main__":
    main()