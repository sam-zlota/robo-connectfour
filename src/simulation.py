from connect_four_env import Connect4Env
from tqdm import trange
import numpy as np

from agent import ExpertAgent, RandomAgent


def simulate_game(first_agent, second_agent):
    env = Connect4Env()
    game_over = False
    episode_len = 0
    while not game_over:

        board, reward, done, _ = env.step(first_agent(env))
        if done:
            return reward, episode_len
        else:
            episode_len += 1

        board, reward, done, _ = env.step(second_agent(env))
        if done:
            return -1 * reward, episode_len
        else:
            episode_len += 1


def simulate_many_games(agent1, agent2, num_games=100, report=False):
    if report:
        print(f'Running {num_games} games between {agent1.get_name()} and {agent2.get_name()}')

    agent1_games_won = 0
    agent2_games_won = 0
    games_drawn = 0
    episode_lengths = []
    for _ in range(num_games // 2):
        reward, episode_len = simulate_game(agent1, agent2)
        episode_lengths.append(episode_len)
        if reward == 1:
            agent1_games_won += 1
        elif reward == -1:
            agent2_games_won += 1
        else:
            games_drawn += 1

    for _ in range(num_games // 2):
        reward, episode_len = simulate_game(agent2, agent1)
        episode_lengths.append(episode_len)
        if reward == 1:
            agent2_games_won += 1
        elif reward == -1:
            agent1_games_won += 1
        else:
            games_drawn += 1
    if report:
        agent_1_report = f'{agent1.get_name()} won {100 * round(agent1_games_won / num_games, 3)} %'
        print(agent_1_report)
        agent_2_report = f'{agent2.get_name()} won {100 * round(agent2_games_won / num_games, 3)} %'
        print(agent_2_report)
        drawn_report = f'Draws occurred {100 * round(games_drawn / num_games, 3)} %  '
        print(drawn_report)
    return agent1_games_won / num_games, agent2_games_won / num_games, episode_lengths


def simulate_against_random(agent, num_games=100):
    _, success_rate, episode_lengths = simulate_many_games(RandomAgent(), agent, num_games=num_games)
    return success_rate, np.mean(episode_lengths)


def simulate_against_expert(agent, num_games=50):
    _, success_rate, episode_lengths = simulate_many_games(ExpertAgent(), agent, num_games=num_games)
    return success_rate, np.mean(episode_lengths)


def random_baseline():
    simulate_many_games(RandomAgent(), RandomAgent(), num_games=1000, report=True)


def expert_baseline():
    simulate_many_games(RandomAgent(), ExpertAgent(), num_games=1000, report=True)


def expert_baseline2():
    simulate_many_games(ExpertAgent(), ExpertAgent(), num_games=3000, report=True)


if __name__ == '__main__':
    # random_baseline()
    # expert_baseline()
    expert_baseline2()
