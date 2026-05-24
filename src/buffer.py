import numpy as np


class RolloutBuffer:
    def __init__(self):
        self.reset()

    def reset(self):
        self.obs, self.acts, self.logps = [], [], []
        self.rews, self.costs, self.dones = [], [], []
        self.val_r, self.val_c = [], []

    def add(self, obs, act, logp, rew, cost, done, val_r, val_c):
        self.obs.append(np.asarray(obs, np.float32))
        self.acts.append(np.asarray(act, np.float32))
        self.logps.append(np.float32(logp))
        self.rews.append(np.float32(rew))
        self.costs.append(np.float32(cost))
        self.dones.append(np.float32(done))
        self.val_r.append(np.float32(val_r))
        self.val_c.append(np.float32(val_c))

    def compute_advantages(self, last_val_r, last_val_c, gamma, lam):
        rews = np.asarray(self.rews, np.float32)
        costs = np.asarray(self.costs, np.float32)
        vals_r = np.asarray(self.val_r + [last_val_r], np.float32)
        vals_c = np.asarray(self.val_c + [last_val_c], np.float32)
        dones = np.asarray(self.dones, np.float32)

        T = len(rews)
        adv_r, adv_c = np.zeros(T, np.float32), np.zeros(T, np.float32)
        gae_r, gae_c = 0.0, 0.0

        for t in reversed(range(T)):
            delta_r = rews[t] + gamma * vals_r[t + 1] * (1 - dones[t]) - vals_r[t]
            delta_c = costs[t] + gamma * vals_c[t + 1] * (1 - dones[t]) - vals_c[t]
            gae_r = delta_r + gamma * lam * (1 - dones[t]) * gae_r
            gae_c = delta_c + gamma * lam * (1 - dones[t]) * gae_c
            adv_r[t], adv_c[t] = gae_r, gae_c

        ret_r = adv_r + np.asarray(self.val_r, np.float32)
        ret_c = adv_c + np.asarray(self.val_c, np.float32)

        adv_r = (adv_r - adv_r.mean()) / (adv_r.std() + 1e-8)
        adv_c = (adv_c - adv_c.mean()) / (adv_c.std() + 1e-8)

        return {
            "obs": np.asarray(self.obs, np.float32),
            "acts": np.asarray(self.acts, np.float32),
            "logps": np.asarray(self.logps, np.float32),
            "adv_r": adv_r, "ret_r": ret_r,
            "adv_c": adv_c, "ret_c": ret_c
        }