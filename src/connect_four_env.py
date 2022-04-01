import gym as gym
import numpy as np
from gym import spaces
from PIL import Image, ImageDraw
import pygame
from numba import jit

from agent import expert_action


class Connect4Env(gym.Env):
    """
    Fills top down as oppsoed to bottom up
    """
    def __init__(self, width=7, height=6, dense_reward=False):
        self.height = height
        self.width = width
        self.board = np.zeros((height, width), dtype="int32")
        self.player = 1

        self.observation_space = spaces.Box(low=-1,
                                            high=1,
                                            shape=(3, height, width),  # fix
                                            dtype=int)
        self.action_space = spaces.Discrete(width)
        self.dense_reward = dense_reward
        if dense_reward:
            self.num_three_in_a_row_p1 = 0
            self.num_two_in_a_row_p1 = 0
            self.num_three_in_a_row_p2 = 0
            self.num_two_in_a_row_p2 = 0

    def to_play(self):
        return 0 if self.player == 1 else 1

    def reset(self, num_pre_moves=0):
        self.board = np.zeros((self.height, self.width), dtype="int32")
        self.player = 1
        for _ in range(num_pre_moves):
            self.step(np.random.choice(self.legal_actions()))
        return self.get_observation()

    def step(self, action):
        for i in range(self.height):
            if self.board[i][action] == 0:
                self.board[i][action] = self.player
                break

        has_winner = have_winner(self.board, self.player)
        done = has_winner or len(self.legal_actions()) == 0
        reward = 1 if has_winner else 0

        if self.dense_reward and reward == 0:
            num_three_in_a_row = any_three(self.board, self.player)
            if self.player == 1:
                if num_three_in_a_row > self.num_three_in_a_row_p1:
                    reward = 0.85
                    self.num_three_in_a_row_p1 = num_three_in_a_row
                else:
                    num_two_in_a_row = any_two(self.board, self.player)
                    if num_two_in_a_row > self.num_two_in_a_row_p1:
                        reward = 0.5
                        self.num_two_in_a_row_p1 = num_two_in_a_row
            else:
                if num_three_in_a_row > self.num_three_in_a_row_p2:
                    reward = 0.85
                    self.num_three_in_a_row_p2 = num_three_in_a_row
                else:
                    num_two_in_a_row = any_two(self.board, self.player)
                    if num_two_in_a_row > self.num_two_in_a_row_p2:
                        reward = 0.1
                        self.num_two_in_a_row_p2 = num_two_in_a_row

        self.player *= -1

        return self.get_observation(), reward, done, []

    def get_observation(self):
        board_player1 = np.where(self.board == 1, 1.0, 0.0)
        board_player2 = np.where(self.board == -1, 1.0, 0.0)
        board_to_play = np.full((6, 7), self.player, dtype="int32")
        return np.array([board_player1, board_player2, board_to_play])

    def legal_actions(self):
        return np.argwhere(self.board[5] == 0).reshape(-1)


@jit
def have_winner(board, player):
    # Horizontal check
    for i in range(4):
        for j in range(6):
            if (
                    board[j][i] == player
                    and board[j][i + 1] == player
                    and board[j][i + 2] == player
                    and board[j][i + 3] == player
            ):
                return True

    # Vertical check
    for i in range(7):
        for j in range(3):
            if (
                    board[j][i] == player
                    and board[j + 1][i] == player
                    and board[j + 2][i] == player
                    and board[j + 3][i] == player
            ):
                return True

    # Positive diagonal check
    for i in range(4):
        for j in range(3):
            if (
                    board[j][i] == player
                    and board[j + 1][i + 1] == player
                    and board[j + 2][i + 2] == player
                    and board[j + 3][i + 3] == player
            ):
                return True

    # Negative diagonal check
    for i in range(4):
        for j in range(3, 6):
            if (
                    board[j][i] == player
                    and board[j - 1][i + 1] == player
                    and board[j - 2][i + 2] == player
                    and board[j - 3][i + 3] == player
            ):
                return True

    return False


@jit
def any_three(board, player):
    amt = 0
    # Horizontal check
    for i in range(5):
        for j in range(6):
            if (
                    board[j][i] == player
                    and board[j][i + 1] == player
                    and board[j][i + 2] == player
                    # and board[j][i + 3] == player
            ):
                amt += 1

                # Vertical check
    for i in range(7):
        for j in range(4):
            if (
                    board[j][i] == player
                    and board[j + 1][i] == player
                    and board[j + 2][i] == player
                    # and board[j + 3][i] == player
            ):
                amt += 1

    # Positive diagonal check
    for i in range(4):
        for j in range(4):
            if (
                    board[j][i] == player
                    and board[j + 1][i + 1] == player
                    and board[j + 2][i + 2] == player
                    # and board[j + 3][i + 3] == player
            ):
                amt += 1

    # Negative diagonal check
    for i in range(5):
        for j in range(3, 6):
            if (
                    board[j][i] == player
                    and board[j - 1][i + 1] == player
                    and board[j - 2][i + 2] == player
                    # and board[j - 3][i + 3] == player
            ):
                amt += 1
    return amt


@jit
def any_two(board, player):
    amt = 0
    # Horizontal check
    for i in range(6):
        for j in range(6):
            if (
                    board[j][i] == player
                    and board[j][i + 1] == player
                    # and board[j][i + 2] == player
                    # and board[j][i + 3] == player
            ):
                amt += 1

    # Vertical check
    for i in range(7):
        for j in range(5):
            if (
                    board[j][i] == player
                    and board[j + 1][i] == player
                    # and board[j + 2][i] == player
                    # and board[j + 3][i] == player
            ):
                amt += 1

    # Positive diagonal check
    for i in range(4):
        for j in range(5):
            if (
                    board[j][i] == player
                    and board[j + 1][i + 1] == player
                    # and board[j + 2][i + 2] == player
                    # and board[j + 3][i + 3] == player
            ):
                amt += 1

    # Negative diagonal check
    for i in range(6):
        for j in range(3, 6):
            if (
                    board[j][i] == player
                    and board[j - 1][i + 1] == player
                    # and board[j - 2][i + 2] == player
                    # and board[j - 3][i + 3] == player
            ):
                amt += 1

    return amt
