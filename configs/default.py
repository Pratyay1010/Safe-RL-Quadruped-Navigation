import torch

CONFIG = {
    "ENV_ID": "SafetyDoggoGoal2-v0",
    "SEED": 0,
    "TIMESTEPS_PER_ROLLOUT": 4096,
    "PPO_EPOCHS": 3,
    "MINI_BATCH_SIZE": 128,
    "CLIP_EPS": 0.2,
    "LR_POLICY": 1e-4,
    "LR_VALUE": 2e-4,
    "ENT_COEF": 1e-3,
    "VF_COEF": 0.5,
    "MAX_GRAD_NORM": 0.5,
    "GAMMA": 0.99,
    "GAE_LAMBDA": 0.95,
    "COST_LIMIT": 20.0,
    "LAMBDA_LR": 1e-3,
    "LAMBDA_INIT": 0.0,
    "TOTAL_UPDATES": 500,
    "SAVE_PATH": "checkpoints/best_model.pth",
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu"
}