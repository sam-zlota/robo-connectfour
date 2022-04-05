import numpy as np

import torch


class AgentOpponent:

    def __init__(self, device, n_actions, cap=5):
        self.networks = [None] * cap
        self.size = 0
        self.cap = cap
        self.device = device
        self.n_actions = n_actions

    def update(self, new_network):
        self.networks[(self.size) % self.cap] = new_network
        self.size += 1
        # print(f'opponent updated {self.size}')

    def act(self, env):
        #return np.min(env.legal_actions())
        #state = env.player * env.board
        t_obs= torch.tensor(env.get_observation(), dtype=torch.float32,
                               device=self.device).unsqueeze(0)
        # acts = {act: 0 for act in range(self.n_actions)}
        #
        # #random oppoenent
        # for net_ndx in range(self.size % self.cap):
        #     network = self.networks[net_ndx]
        #     pred = network.predict(t_state).item()
        #     acts[pred] += 1
        # return max(acts, key=acts.get)
        # acts = {act: 0 for act in range(self.n_actions)}

        # random oppoenent
        net_ndx = np.random.choice(np.arange(min(self.size, self.cap)))
        network = self.networks[net_ndx]
        pred = network.predict(t_obs, env)
        return pred
