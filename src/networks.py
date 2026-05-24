import torch
import torch.nn as nn


LOG_STD_MIN = -20
LOG_STD_MAX = 2
EPS = 1e-8


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )

        self.mean = nn.Linear(256, act_dim)
        self.log_std = nn.Parameter(
            torch.zeros(act_dim, dtype=torch.float32)
        )

    def forward(self, obs):
        features = self.net(obs)

        return (
            self.mean(features),
            torch.clamp(self.log_std, LOG_STD_MIN, LOG_STD_MAX)
        )

    def get_action_and_logp(self, obs):
        mean, log_std = self.forward(obs)

        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mean, std)

        latent_action = dist.rsample()
        action = torch.tanh(latent_action)

        logp = (
            dist.log_prob(latent_action).sum(-1)
            - torch.log(1 - action.pow(2) + EPS).sum(-1)
        )

        return action, logp

    def act(self, obs):
        with torch.no_grad():
            action, _ = self.get_action_and_logp(obs)

        return action.cpu().numpy().squeeze()


class Critic(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, obs):
        return self.net(obs).squeeze(-1)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()

        self.actor = Actor(obs_dim, act_dim)

        # Separate critics for reward and safety cost estimation
        self.reward_critic = Critic(obs_dim)
        self.cost_critic = Critic(obs_dim)