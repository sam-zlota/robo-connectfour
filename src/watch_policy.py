import time

import numpy as np
import pygame
import pygame.gfxdraw
import random
import enum
from copy import deepcopy
import os

# Graphical size settings
from agent import RandomAgent, ExpertAgent

from connect_four_env import Connect4Env

from dqn_agent import DQNAgent

from render import Connect4Game, Connect4Viewer, SQUARE_SIZE


def load_ai_agent(path, env):
    agent = DQNAgent(env,
                     gamma=0.8,
                     learning_rate=5e-4,
                     buffer_size=10_000,
                     initial_epsilon=0.99,
                     final_epsilon=0.70,
                     exploration_fraction=0.9,
                     target_network_update_freq=100,  # temproaly correlated epsiode?
                     batch_size=256,
                     device='cpu',
                     update_method='standard'
                     )
    agent.load(path)
    return agent


def watch_self_play(path):
    game = Connect4Game()
    env = Connect4Env()
    game.reset_game()

    view = Connect4Viewer(game=game)
    view.initialize()
    dqn_agent = load_ai_agent(path, env)
    # dqn_agent = load_ai_agent(
    #     '/Users/szlota777/Desktop/Spring2022/CS4910/connect_four/robo-connectfour/saved/20220403-212644/network_weights', env)
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if game.get_win() is None:
                print('DQN Running')
                computer_act = dqn_agent(env)
                print('DQN took ', computer_act)
                env.step(computer_act)
                game.place(computer_act)
                print(env.board[::-1], env.legal_actions())
                time.sleep(1)
            else:
                game.reset_game()
                env.reset()


def watch_expert_play():
    game = Connect4Game()
    env = Connect4Env()
    game.reset_game()

    view = Connect4Viewer(game=game)
    view.initialize()
    expert_agent = ExpertAgent()
    # dqn_agent = load_ai_agent(
    #     '/Users/szlota777/Desktop/Spring2022/CS4910/connect_four/robo-connectfour/saved/20220403-212644/network_weights', env)
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if game.get_win() is None:
                expert_act = expert_agent(env)
                env.step(expert_act)
                game.place(expert_act)
                print(env.board[::-1], env.legal_actions())
                time.sleep(1)
            else:
                game.reset_game()
                env.reset()


def watch_play_against_expert(path):
    game = Connect4Game()
    env = Connect4Env()
    game.reset_game()

    view = Connect4Viewer(game=game)
    view.initialize()

    dqn_agent = load_ai_agent(path, env)
    # rand_agent = RandomAgent()
    expert_agent = ExpertAgent()
    # dqn_agent = load_ai_agent(
    #     '/Users/szlota777/Desktop/Spring2022/CS4910/connect_four/robo-connectfour/saved/20220403-212644/network_weights', env)
    running = True
    expert_turn = True
    while running:
        for event in pygame.event.get():
            print(env.board[::-1], env.legal_actions())
            if event.type == pygame.QUIT:
                running = False
            if expert_turn:
                if game.get_win() is None:
                    expert_act = expert_agent(env)
                    env.step(expert_act)
                    game.place(expert_act)
                    expert_turn = False
                    time.sleep(1)
                else:
                    game.reset_game()
                    env.reset()
                    expert_turn = True
            else:
                if game.get_win() is None:
                    print('Computer Running')
                    computer_act = dqn_agent(env)
                    print('Computer took', computer_act)
                    env.step(computer_act)
                    game.place(computer_act)
                    expert_turn = True
                    time.sleep(1)
                else:
                    game.reset_game()
                    env.reset()
                    expert_turn = True


def human_play_against(path):
    game = Connect4Game()
    env = Connect4Env()
    dqn_agent = load_ai_agent(path, env)
    game.reset_game()

    view = Connect4Viewer(game=game)
    view.initialize()

    running = True
    human_turn = False
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if human_turn:

                if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                    if game.get_win() is None:
                        human_act = pygame.mouse.get_pos()[0] // SQUARE_SIZE
                        env.step(human_act)
                        game.place(human_act)
                        human_turn = False
                    else:
                        game.reset_game()
                        env.reset()
                        human_turn = False

            else:
                if game.get_win() is None:
                    print('Computer Running')
                    computer_act = dqn_agent(env)
                    env.step(computer_act)
                    game.place(computer_act)
                    print('Computer took', computer_act)
                    human_turn = True
                else:
                    game.reset_game()
                    env.reset()
                    human_turn = True


def human_play_against_expert():
    game = Connect4Game()
    env = Connect4Env()
    expert = ExpertAgent()
    game.reset_game()

    view = Connect4Viewer(game=game)
    view.initialize()

    running = True
    human_turn = False
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if human_turn:

                if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                    if game.get_win() is None:
                        human_act = pygame.mouse.get_pos()[0] // SQUARE_SIZE
                        env.step(human_act)
                        game.place(human_act)
                        human_turn = False
                    else:
                        game.reset_game()
                        env.reset()
                        human_turn = False

            else:
                if game.get_win() is None:
                    print('Expert Running')
                    computer_act = expert(env)
                    env.step(computer_act)
                    game.place(computer_act)
                    print('Expert took', computer_act)
                    human_turn = True
                else:
                    game.reset_game()
                    env.reset()
                    human_turn = True


if __name__ == '__main__':
    save_dir = '/Users/szlota777/Desktop/Spring2022/CS4910/connect_four/robo-connectfour/saved/20220407-231913'
    watch_self_play(os.path.join(save_dir, 'final_network_weights'))
    # human_play_against_expert()
