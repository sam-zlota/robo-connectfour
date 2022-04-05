import torch.nn as nn
from torch import Tensor
import torch
import numpy as np


class QNetwork(nn.Module):
    def __init__(self, )  -> None:
        '''Q-Network instantiated as 3-layer MLP with 64 units

        Parameters
        ----------
        state_dim
            length of state vector
        n_actions
            number of actions in action space
        '''
        super().__init__()
        observation_size = (3 * 6 * 7)
        n_actions = 7
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(observation_size, 256),
            nn.ReLU(True),
            nn.Linear(256, 256),
            nn.ReLU(True),
            nn.Linear(256, 64),
            nn.ReLU(True),
            nn.Linear(64, n_actions)
        )

        self.loss_fn = nn.MSELoss()

    def forward(self, s: Tensor) -> Tensor:
        '''Perform forward pass

        Parameters
        ----------
        s
            state tensor; shape=(B,|S|); dtype=float32
        Returns
        -------
        tensor of q values for each action; shape=(B,|A|); dtype=float32
        '''
        # print(s.shape)
        return self.layers(s)

    @torch.no_grad()
    def predict(self, s: Tensor, env) -> Tensor:
        '''Computes argmax over q-function at given states
        '''

        q_vals = self.forward(s)
        best_acts = torch.argsort(q_vals, descending=True).reshape(-1).tolist()

        # only choose valid actions
        for act in best_acts:
            if env.board[5][act] == 0:
                return act

        return -1

    def compute_loss(self, q_pred: Tensor, q_target: Tensor) -> Tensor:
        return self.loss_fn(q_pred, q_target)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
