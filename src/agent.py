import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from networks import ActorCritic
from buffer import RolloutBuffer


class Agent:
    def __init__(self, env, config):
        self.env = env
        self.cfg = config
        self.device = torch.device(self.cfg["DEVICE"])

        obs_dim = int(np.prod(env.observation_space.shape))
        act_dim = int(np.prod(env.action_space.shape))

        self.ac = ActorCritic(obs_dim, act_dim).to(self.device)

        self.opt_pi = optim.Adam(
            self.ac.actor.parameters(),
            lr=self.cfg["LR_POLICY"]
        )

        self.opt_vr = optim.Adam(
            self.ac.reward_critic.parameters(),
            lr=self.cfg["LR_VALUE"]
        )

        self.opt_vc = optim.Adam(
            self.ac.cost_critic.parameters(),
            lr=self.cfg["LR_VALUE"]
        )

        self.cost_penalty_coef = float(self.cfg["LAMBDA_INIT"])

        self.buffer = RolloutBuffer()

        self.best_reward = -np.inf
        self.best_model = None

    def load_model(self, path):
        if not os.path.exists(path):
            print(f"Model file not found: {path}")
            return False

        checkpoint = torch.load(
            path,
            map_location=self.device,
            weights_only=False
        )

        self.ac.actor.load_state_dict(checkpoint["actor"])

        self.ac.reward_critic.load_state_dict(
            checkpoint["reward_critic"]
        )

        self.ac.cost_critic.load_state_dict(
            checkpoint["cost_critic"]
        )

        self.cost_penalty_coef = checkpoint["cost_penalty_coef"]

        print(f"Loaded model from {path}")

        return True

    def get_action(self, obs):
        obs_t = torch.as_tensor(
            obs,
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(0)

        return self.ac.actor.act(obs_t)

    def estimate_values(self, obs):
        with torch.no_grad():
            obs_t = torch.as_tensor(
                np.asarray(obs, np.float32),
                device=self.device
            )

            reward_value = float(
                self.ac.reward_critic(obs_t).cpu()
            )

            cost_value = float(
                self.ac.cost_critic(obs_t).cpu()
            )

        return reward_value, cost_value

    def collect_rollout(self):
        self.buffer.reset()

        timesteps = 0
        ep_rewards, ep_costs = [], []

        # Collect on-policy trajectories for PPO updates
        while timesteps < self.cfg["TIMESTEPS_PER_ROLLOUT"]:

            obs, _ = self.env.reset()

            done = False

            ep_reward = 0.0
            ep_cost = 0.0

            while (
                not done
                and timesteps < self.cfg["TIMESTEPS_PER_ROLLOUT"]
            ):

                obs_t = torch.as_tensor(
                    obs,
                    dtype=torch.float32,
                    device=self.device
                ).unsqueeze(0)

                with torch.no_grad():
                    action_t, logp_t = (
                        self.ac.actor.get_action_and_logp(obs_t)
                    )

                    reward_value = (
                        self.ac.reward_critic(obs_t)
                        .cpu()
                        .numpy()
                        .squeeze()
                    )

                    cost_value = (
                        self.ac.cost_critic(obs_t)
                        .cpu()
                        .numpy()
                        .squeeze()
                    )

                action = action_t.cpu().numpy().squeeze()

                step = self.env.step(action)

                if len(step) == 6:
                    (
                        next_obs,
                        reward,
                        cost,
                        terminated,
                        truncated,
                        info,
                    ) = step
                else:
                    (
                        next_obs,
                        reward,
                        terminated,
                        truncated,
                        info,
                    ) = step

                    cost = info.get("cost", 0.0)

                done = bool(terminated or truncated)

                timesteps += 1

                ep_reward += reward
                ep_cost += cost

                logp_value = float(
                    logp_t.cpu().numpy().item()
                )

                self.buffer.add(
                    obs,
                    action,
                    logp_value,
                    reward,
                    cost,
                    done,
                    reward_value,
                    cost_value,
                )

                obs = next_obs

            ep_rewards.append(ep_reward)
            ep_costs.append(ep_cost)

        last_reward_value, last_cost_value = (
            self.estimate_values(obs)
        )

        return (
            last_reward_value,
            last_cost_value,
            ep_rewards,
            ep_costs,
        )

    def update(self, data):

        def to_tensor(x):
            return torch.as_tensor(
                x,
                dtype=torch.float32,
                device=self.device
            )

        obs = to_tensor(data["obs"])
        acts = to_tensor(data["acts"])
        old_logps = to_tensor(data["logps"])

        reward_adv = to_tensor(data["adv_r"])
        reward_returns = to_tensor(data["ret_r"])

        cost_adv = to_tensor(data["adv_c"])
        cost_returns = to_tensor(data["ret_c"])

        n = obs.shape[0]

        indices = np.arange(n)

        for _ in range(self.cfg["PPO_EPOCHS"]):

            np.random.shuffle(indices)

            for start in range(
                0,
                n,
                self.cfg["MINI_BATCH_SIZE"]
            ):

                mb = indices[
                    start:start + self.cfg["MINI_BATCH_SIZE"]
                ]

                o = obs[mb]
                a = acts[mb]
                old_lp = old_logps[mb]

                mean, log_std = self.ac.actor(o)

                std = torch.exp(log_std)

                def atanh(x):
                    return 0.5 * (
                        torch.log1p(x)
                        - torch.log1p(-x)
                    )

                latent_action = atanh(
                    torch.clamp(a, -0.999, 0.999)
                )

                logp_u = (
                    torch.distributions.Normal(mean, std)
                    .log_prob(latent_action)
                    .sum(-1)
                )

                logp = (
                    logp_u
                    - torch.log(
                        1 - torch.tanh(latent_action).pow(2)
                        + 1e-8
                    ).sum(-1)
                )

                ratio = torch.exp(logp - old_lp)

                surr1 = ratio * reward_adv[mb]

                surr2 = (
                    torch.clamp(
                        ratio,
                        1 - self.cfg["CLIP_EPS"],
                        1 + self.cfg["CLIP_EPS"]
                    )
                    * reward_adv[mb]
                )

                policy_loss = -torch.mean(
                    torch.min(surr1, surr2)
                )

                entropy = torch.mean(
                    0.5 * (
                        log_std * 2
                        + 1.8378770664093453
                    )
                )

                entropy_loss = (
                    -self.cfg["ENT_COEF"] * entropy
                )

                cost_penalty = (
                    self.cost_penalty_coef
                    * torch.mean(ratio * cost_adv[mb])
                )

                total_loss = (
                    policy_loss
                    + cost_penalty
                    + entropy_loss
                )

                self.opt_pi.zero_grad()

                total_loss.backward()

                nn.utils.clip_grad_norm_(
                    self.ac.actor.parameters(),
                    self.cfg["MAX_GRAD_NORM"]
                )

                self.opt_pi.step()

                reward_pred = self.ac.reward_critic(o)
                cost_pred = self.ac.cost_critic(o)

                reward_loss = nn.MSELoss()(
                    reward_pred,
                    reward_returns[mb]
                )

                cost_loss = nn.MSELoss()(
                    cost_pred,
                    cost_returns[mb]
                )

                self.opt_vr.zero_grad()

                (
                    self.cfg["VF_COEF"]
                    * reward_loss
                ).backward()

                self.opt_vr.step()

                self.opt_vc.zero_grad()

                (
                    self.cfg["VF_COEF"]
                    * cost_loss
                ).backward()

                self.opt_vc.step()

    def update_penalty_coef(self, mean_cost):
        diff = mean_cost - self.cfg["COST_LIMIT"]

        self.cost_penalty_coef = max(
            0.0,
            self.cost_penalty_coef
            + self.cfg["LAMBDA_LR"] * diff
        )

    def train(self):
        np.random.seed(self.cfg["SEED"])
        torch.manual_seed(self.cfg["SEED"])

        total_steps = 0

        save_dir = os.path.dirname(
            self.cfg["SAVE_PATH"]
        )

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        for update in range(
            1,
            self.cfg["TOTAL_UPDATES"] + 1
        ):

            (
                last_reward_value,
                last_cost_value,
                ep_rewards,
                ep_costs,
            ) = self.collect_rollout()

            total_steps += len(self.buffer.rews)

            data = self.buffer.compute_advantages(
                last_reward_value,
                last_cost_value,
                self.cfg["GAMMA"],
                self.cfg["GAE_LAMBDA"]
            )

            self.update(data)

            mean_reward = np.mean(ep_rewards)
            mean_cost = np.mean(ep_costs)

            self.update_penalty_coef(mean_cost)

            print(
                f"[Update {update}] "
                f"steps={total_steps} "
                f"reward={mean_reward:.3f} "
                f"cost={mean_cost:.3f} "
                f"penalty={self.cost_penalty_coef:.4f}"
            )

            model_data = {
                "actor": self.ac.actor.state_dict(),
                "reward_critic": self.ac.reward_critic.state_dict(),
                "cost_critic": self.ac.cost_critic.state_dict(),
                "cost_penalty_coef": self.cost_penalty_coef,
            }

            if mean_reward > self.best_reward:
                self.best_reward = mean_reward
                self.best_model = model_data

            if (
                update % 5 == 0
                and self.best_model is not None
            ):

                torch.save(
                    self.best_model,
                    self.cfg["SAVE_PATH"]
                )

                print(
                    f"Saved best model "
                    f"(reward={self.best_reward:.3f})"
                )

        if self.best_model is not None:

            torch.save(
                self.best_model,
                self.cfg["SAVE_PATH"]
            )

            print(
                f"Saved final best model "
                f"(reward={self.best_reward:.3f})"
            )

        self.env.close()