import torch.nn as nn
from torch import Tensor
import torch
import numpy as np


class QNetwork(nn.Module):
    def __init__(self, state_dim, n_actions: int) -> None:
        '''Q-Network instantiated as 3-layer MLP with 64 units

        Parameters
        ----------
        state_dim
            length of state vector
        n_actions
            number of actions in action space
        '''
        super().__init__()
        # print(state_dim)
        self.layers = nn.Sequential(
            # nn.Conv2d(in_channels=3, out_channels=8, kernel_size=(3, 3), padding=1),
            # nn.BatchNorm2d(8),
            # nn.ReLU(True),
            # nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=1),
            # nn.BatchNorm2d(16),
            # nn.ReLU(True),
            # nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=1),
            # nn.BatchNorm2d(32),
            # nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(3*42, 256),
            nn.ReLU(True),
            nn.Linear(256, 256),
            nn.ReLU(True),
            nn.Linear(256, 64),
            nn.ReLU(True),
            nn.Linear(64, n_actions)
        )

        self.loss_fn = nn.MSELoss()

    def conv_out_size(self):
        random_in = torch.tensor(np.zeros((3, 6, 7)), dtype=torch.float32).unsqueeze(0)
        # print(random_in.shape)
        conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
        )
        # print(conv_layers(random_in).shape)
        # print(tuple(list(conv_layers(random_in).shape)))
        return np.prod(np.array((conv_layers(random_in).shape)))

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

        for act in best_acts:

            if env.board[5][act] == 0:
                return act

        print(q_vals)
        print(best_acts, env.legal_actions())
        return -1

    def compute_loss(self, q_pred: Tensor, q_target: Tensor) -> Tensor:
        return self.loss_fn(q_pred, q_target)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
        # self.eval()
