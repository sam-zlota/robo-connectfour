import numpy as np
from typing import Tuple


class ReplayBuffer:
    def __init__(self, size: int, state_dim: int) -> None:
        self.data = {'s': np.zeros((size, 3, 6, 7), dtype=np.float32),
                     'a': np.zeros((size), dtype=np.int32),
                     'r': np.zeros((size), dtype=np.float32),
                     'sp': np.zeros((size, 3, 6, 7), dtype=np.float32),
                     'd': np.zeros((size), dtype=np.bool8),
                     }

        self.size = size
        self.length = 0
        self._idx = 0

    def add_transition(self, s: np.ndarray, a: int, r: float,
                       sp: np.ndarray, d: bool) -> None:
        self.data['s'][self._idx] = s
        self.data['a'][self._idx] = a
        self.data['r'][self._idx] = r
        self.data['sp'][self._idx] = sp
        self.data['d'][self._idx] = d

        self._idx = (self._idx + 1) % self.size

        self._idx = (self._idx + 1) % self.size
        self.length = min(self.length + 1, self.size)

    def sample(self, batch_size: int) -> Tuple:
        idxs = np.random.randint(0, self.length, batch_size)

        s = self.data['s'][idxs]
        a = self.data['a'][idxs]
        r = self.data['r'][idxs]
        sp = self.data['sp'][idxs]
        d = self.data['d'][idxs]

        return s, a, r, sp, d
