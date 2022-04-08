import numpy as np

import torch


class AgentOpponent:
    """ A memory of past selves. Chooses actions based on random seletion past q network."""

    def __init__(self, device, n_actions, cap=5):
        self.networks = [None] * cap
        self.size = 0
        self.cap = cap
        self.device = device
        self.n_actions = n_actions

    def update(self, new_network):
        self.networks[self.size % self.cap] = new_network
        self.size += 1

    def act(self, env):
        t_obs = torch.tensor(env.get_observation(), dtype=torch.float32,
                             device=self.device).unsqueeze(0)

        # random oppoenent
        net_ndx = np.random.choice(np.arange(min(self.size, self.cap)))
        network = self.networks[net_ndx]
        pred = network.predict(t_obs, env)
        return pred
