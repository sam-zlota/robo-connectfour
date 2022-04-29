import os
from datetime import datetime

from agent import RandomAgent, ExpertAgent
from connect_four_env import Connect4Env
import matplotlib.pyplot as plt
import numpy as np


class SuccessTracker:

    def __init__(self):
        self.evaluation_metrics = {
            # agent vs. random
            'agent_vs_random_wins_percentage': [],
            'agent_vs_random_losses_percentage': [],
            'agent_vs_random_draws_percentage': [],
            'agent_vs_random_episode_length_average': [],
            # random vs. agent
            'random_vs_agent_wins_percentage': [],
            'random_vs_agent_losses_percentage': [],
            'random_vs_agent_draws_percentage': [],
            'random_vs_agent_episode_length_average': [],
            # agent vs. expert
            'agent_vs_expert_wins_percentage': [],
            'agent_vs_expert_losses_percentage': [],
            'agent_vs_expert_draws_percentage': [],
            'agent_vs_expert_episode_length_average': [],
            # expert vs. agent
            'expert_vs_agent_wins_percentage': [],
            'expert_vs_agent_losses_percentage': [],
            'expert_vs_agent_draws_percentage': [],
            'expert_vs_agent_episode_length_average': [],
            # agent vs. agent , perspective of first player
            'agent_vs_agent_wins_percentage': [],
            'agent_vs_agent_losses_percentage': [],
            'agent_vs_agent_draws_percentage': [],
            'agent_vs_agent_episode_length_average': [],
        }

    def evaluate(self, agent):
        self.evaluate_agent_vs_random(agent)
        self.evaluate_agent_vs_expert(agent)
        self.evaluate_random_vs_agent(agent)
        self.evaluate_expert_vs_agent(agent)
        self.evaluate_agent_vs_agent(agent)

    def evaluate_agent_vs_random(self, agent):
        win_percentage, loss_percentage, draw_percentage, average_game_length = self.play_many_games(agent,
                                                                                                     RandomAgent())
        self.evaluation_metrics['agent_vs_random_wins_percentage'].append(win_percentage)
        self.evaluation_metrics['agent_vs_random_losses_percentage'].append(loss_percentage)
        self.evaluation_metrics['agent_vs_random_draws_percentage'].append(draw_percentage)
        self.evaluation_metrics['agent_vs_random_episode_length_average'].append(average_game_length)

    def evaluate_random_vs_agent(self, agent):
        loss_percentage, win_percentage, draw_percentage, average_game_length = self.play_many_games(RandomAgent(),
                                                                                                     agent)
        self.evaluation_metrics['random_vs_agent_wins_percentage'].append(win_percentage)
        self.evaluation_metrics['random_vs_agent_losses_percentage'].append(loss_percentage)
        self.evaluation_metrics['random_vs_agent_draws_percentage'].append(draw_percentage)
        self.evaluation_metrics['random_vs_agent_episode_length_average'].append(average_game_length)

    def evaluate_agent_vs_expert(self, agent):
        win_percentage, loss_percentage, draw_percentage, average_game_length = self.play_many_games(agent,
                                                                                                     ExpertAgent())
        self.evaluation_metrics['agent_vs_expert_wins_percentage'].append(win_percentage)
        self.evaluation_metrics['agent_vs_expert_losses_percentage'].append(loss_percentage)
        self.evaluation_metrics['agent_vs_expert_draws_percentage'].append(draw_percentage)
        self.evaluation_metrics['agent_vs_expert_episode_length_average'].append(average_game_length)

    def evaluate_expert_vs_agent(self, agent):
        loss_percentage, win_percentage, draw_percentage, average_game_length = self.play_many_games(ExpertAgent(),
                                                                                                     agent)
        self.evaluation_metrics['expert_vs_agent_wins_percentage'].append(win_percentage)
        self.evaluation_metrics['expert_vs_agent_losses_percentage'].append(loss_percentage)
        self.evaluation_metrics['expert_vs_agent_draws_percentage'].append(draw_percentage)
        self.evaluation_metrics['expert_vs_agent_episode_length_average'].append(average_game_length)

    def evaluate_agent_vs_agent(self, agent):
        win_percentage, loss_percentage, draw_percentage, average_game_length = self.play_many_games(agent,
                                                                                                     agent, 1)
        self.evaluation_metrics['agent_vs_agent_wins_percentage'].append(win_percentage)
        self.evaluation_metrics['agent_vs_agent_losses_percentage'].append(loss_percentage)
        self.evaluation_metrics['agent_vs_agent_draws_percentage'].append(draw_percentage)
        self.evaluation_metrics['agent_vs_agent_episode_length_average'].append(average_game_length)

    @staticmethod
    def play_game(first_player, second_player):
        env = Connect4Env(simulation_mode=True)
        game_over = False
        num_steps = 0
        while not game_over:
            first_player_action = first_player(env)
            _, _, game_over, _ = env.step(first_player_action)
            num_steps += 1
            if not game_over:
                second_player_action = second_player(env)
                _, _, game_over, _ = env.step(second_player_action)
                num_steps += 1
        return env.result, num_steps

    @staticmethod
    def play_many_games(player1, player2, num_trials=100.):
        num_wins = 0
        num_losses = 0
        num_draws = 0
        num_steps_sum = 0
        for _ in range(int(num_trials)):
            result, num_steps = SuccessTracker.play_game(player1, player2)
            if result == 1:
                num_wins += 1
            elif result == -1:
                num_losses += 1
            else:
                num_draws += 1
            num_steps_sum += num_steps
        win_percentage = num_wins / num_trials
        loss_percentage = num_losses / num_trials
        draw_percentage = num_draws / num_trials
        average_game_length = num_steps_sum / num_trials
        return win_percentage, loss_percentage, draw_percentage, average_game_length

    def plot_metrics(self, save_path):
        f, axs = plt.subplots(4, 2, figsize=(20, 10))
        axs = axs.flat
        axs[0].plot(self.evaluation_metrics['agent_vs_random_wins_percentage'], label='random', color='green')
        axs[0].plot(self.evaluation_metrics['agent_vs_expert_wins_percentage'], label='expert', color='darkorange')
        # axs[0].plot(self.evaluation_metrics['agent_vs_agent_wins_percentage'], label='self', color='red')
        axs[0].set_title('Win Percentage When Agent Goes First')
        axs[0].legend()

        axs[2].plot(self.evaluation_metrics['agent_vs_random_losses_percentage'], label='random', color='green')
        axs[2].plot(self.evaluation_metrics['agent_vs_expert_losses_percentage'], label='expert', color='darkorange')
        # axs[2].plot(self.evaluation_metrics['agent_vs_agent_losses_percentage'], label='self', color='red')
        axs[2].set_title('Loss Percentage When Agent Goes First')
        axs[2].legend()

        axs[4].plot(self.evaluation_metrics['agent_vs_random_draws_percentage'], label='random', color='green')
        axs[4].plot(self.evaluation_metrics['agent_vs_expert_draws_percentage'], label='expert', color='darkorange')
        # axs[4].plot(self.evaluation_metrics['agent_vs_agent_draws_percentage'], label='self', color='red')
        axs[4].set_title('Draw Percentage When Agent Goes First')
        axs[4].legend()

        axs[6].plot(self.evaluation_metrics['agent_vs_random_episode_length_average'], label='random', color='green')
        axs[6].plot(self.evaluation_metrics['agent_vs_expert_episode_length_average'], label='expert',
                    color='darkorange')
        axs[6].plot(self.evaluation_metrics['agent_vs_agent_episode_length_average'], label='self', color='red')
        axs[6].set_title('Average Game Length When Agent Goes First')
        axs[6].legend()

        axs[1].plot(self.evaluation_metrics['random_vs_agent_wins_percentage'], label='random', color='green')
        axs[1].plot(self.evaluation_metrics['expert_vs_agent_wins_percentage'], label='expert', color='darkorange')
        # axs[1].plot(1 - np.array(self.evaluation_metrics['agent_vs_agent_wins_percentage']), label='self', color='red')
        axs[1].set_title('Win Percentage When Agent Goes Second')
        axs[1].legend()

        axs[3].plot(self.evaluation_metrics['random_vs_agent_losses_percentage'], label='random', color='green')
        axs[3].plot(self.evaluation_metrics['expert_vs_agent_losses_percentage'], label='expert', color='darkorange')
        # axs[3].plot(1 - np.array(self.evaluation_metrics['agent_vs_agent_losses_percentage']), label='self',
        #             color='red')
        axs[3].set_title('Loss Percentage When Agent Goes Second')
        axs[3].legend()

        axs[5].plot(self.evaluation_metrics['random_vs_agent_draws_percentage'], label='random', color='green')
        axs[5].plot(self.evaluation_metrics['expert_vs_agent_draws_percentage'], label='expert', color='darkorange')
        # TODO verify logic
        #axs[5].plot(self.evaluation_metrics['agent_vs_agent_draws_percentage'], label='self', color='red')
        axs[5].set_title('Draw Percentage When Agent Goes Second')
        axs[5].legend()

        axs[7].plot(self.evaluation_metrics['random_vs_agent_episode_length_average'], label='random', color='green')
        axs[7].plot(self.evaluation_metrics['expert_vs_agent_episode_length_average'], label='expert',
                    color='darkorange')
        axs[7].plot(self.evaluation_metrics['agent_vs_agent_episode_length_average'], label='self', color='red')
        axs[7].set_title('Average Game Length When Agent Goes Second')
        axs[7].legend()

        plt.tight_layout()
        plt.savefig(os.path.join(save_path, datetime.now().strftime("%Y%m%d-%H%M%S") + '-evaluation-fig.png'))

        plt.close('all')
