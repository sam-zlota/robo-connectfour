import numpy as np
from numba import jit


class Agent:

    def select_action(self, env):
        raise NotImplementedError

    def get_name(self):
        raise NotImplementedError


class RandomAgent(Agent):
    def select_action(self, env):
        return env.action_space.sample()

    def get_name(self):
        return 'Random Agent'

    __call__ = select_action


class ExpertAgent(Agent):
    def select_action(self, env):
        board = env.board
        player = env.player
        action = np.random.choice(env.legal_actions())
        return expert_action(board, player, action)

    def get_name(self):
        return 'Expert Agent'

    __call__ = select_action


def expert_action(board, player, action):
    for k in np.arange(3):
        for l in np.arange(4):
            sub_board = board[k: k + 4, l: l + 4]
            for i in np.arange(4):
                if np.abs(np.sum(sub_board[i, :])) == 3:
                    ind = np.where(sub_board[i, :] == 0)[0][0]
                    if np.count_nonzero(board[:, ind + l]) == i + k:
                        action = ind + l
                        if player * np.sum(sub_board[i, :]) > 0:
                            return action

                if np.abs(np.sum(sub_board[:, i])) == 3:
                    action = i + l
                    if player * np.sum(sub_board[:, i]) > 0:
                        return action
            # Diagonal checks
            diag = sub_board.diagonal()
            anti_diag = np.fliplr(sub_board).diagonal()
            if np.abs(np.sum(diag)) == 3:
                ind = np.where(diag == 0)[0][0]
                if np.count_nonzero(board[:, ind + l]) == ind + k:
                    action = ind + l
                    if player * np.sum(diag) > 0:
                        return action

            if np.abs(np.sum(anti_diag)) == 3:
                ind = np.where(anti_diag == 0)[0][0]
                if np.count_nonzero(board[:, 3 - ind + l]) == ind + k:
                    action = 3 - ind + l
                    if player * np.sum(anti_diag) > 0:
                        return action

    return action

    print(expert_baseline())
    print(random_baseline())
