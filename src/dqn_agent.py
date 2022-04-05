import os
from datetime import datetime

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm as tqdm

import gym
import numpy as np

from connect_four_env import Connect4Env
from opponent import AgentOpponent
from q_network import QNetwork
from replay_buffer import ReplayBuffer

from simulation import simulate_against_random, simulate_against_expert

from agent import expert_action

import time

"""
https://github.com/IASIAI/gym-connect-four/blob/fbb504596ff868acaf909b29b4f52f0cb0dd6e1e/gym_connect_four/envs/render.py
"""


class SuccessTracker:
    pass


class DQNAgent:
    def __init__(self,
                 env: gym.Env,
                 gamma: float = 1.,
                 learning_rate: float = 5e-4,
                 buffer_size: int = 50000,
                 batch_size: int = 128,
                 initial_epsilon: float = 1.,
                 final_epsilon: float = 0.15,
                 exploration_fraction: float = 0.9,
                 target_network_update_freq: int = 1000,
                 seed: int = 0,
                 device: str = 'cpu',
                 update_method: str = 'standard',
                 plotting_smoothing: int = 200
                 ) -> None:
        '''Agent that learns policy using DQN algorithm
        '''
        self.env = env

        assert 0 < gamma <= 1., 'Discount factor (gamma) must be in range (0,1]'
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_network_update_freq = target_network_update_freq
        self.initial_epsilon = initial_epsilon
        self.final_epsilon = final_epsilon
        self.exploration_fraction = exploration_fraction

        self.buffer = ReplayBuffer(buffer_size, self.env.observation_space.shape[0])

        self.device = device
        self.network = QNetwork(self.env.observation_space.shape[0],
                                self.env.action_space.n).to(device)
        self.target_network = QNetwork(self.env.observation_space.shape,
                                       self.env.action_space.n).to(device)
        self.hard_target_update()

        self.optim = torch.optim.Adam(self.network.parameters(),
                                      lr=learning_rate)

        # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim, step_size=300_000, gamma=0.5)

        self.opponent = AgentOpponent(self.device, self.env.action_space.n)
        self.opponent.update(self.target_network)

        self.update_method = update_method
        self.plotting_smoothing = plotting_smoothing
        np.random.seed(seed)
        torch.manual_seed(seed)
        if device == 'cuda':
            torch.cuda.manual_seed(seed)

    def train(self, num_steps: int, save_path) -> None:
        '''Trains q-network for given number of environment steps, plots
        rewards and loss curve
        '''
        rewards_data = []
        loss_data = []

        success_data_rand = []
        episode_len_rand = []

        success_data_exp = []
        episode_len_exp = []

        training_episode_lengths = []

        best_network = None
        episode_count = 0
        episode_rewards = 0
        opt_count = 0
        s = self.env.reset()

        pbar = tqdm(range(1, num_steps + 1))
        episode_len = 0
        for step in pbar:
            epsilon = self.compute_epsilon(step / (self.exploration_fraction * num_steps))
            # epsilon = max(epsilon, 0.20) #TODO fix this

            a = self._select_action(s, epsilon)
            sp, r, done, info = self.env.step(a)
            episode_len += 1
            # if not done:
            #     # a_opp = self._select_action(sp1, epsilon)
            #     a_opp = self._select_opp_action(epsilon)
            #     sp, r_opp, done, info = self.env.step(a_opp)

            #     r = -1 * r_opp
            #     episode_len += 1

            episode_rewards += r

            self.buffer.add_transition(s=s, a=a, r=r, sp=sp, d=done)

            # optimize
            if self.buffer.length > self.batch_size:
                loss = self.optimize()
                opt_count += 1
                loss_data.append(loss)

                if opt_count % self.target_network_update_freq == 0:
                    self.hard_target_update()
                    # self.opponent.update(self.network)
            if step % 5000 == 0:
                self.opponent.update(self.network)
            # evaluate
            if step % 2000 == 0:
                # TODO expert success rate
                success_rate_rand, elen_rand = simulate_against_random(self)
                success_data_rand.append(success_rate_rand)
                episode_len_rand.append(elen_rand)
                success_rate_exp, elen_exp = simulate_against_expert(self)
                if success_rate_exp > max(success_data_exp, default=0):
                    best_network = self.network
                    # self.opponent.update(self.network)
                success_data_exp.append(success_rate_exp)
                episode_len_exp.append(elen_exp)
                pbar.set_description(f'Success = {success_rate_rand, success_rate_exp}')

                if step % 20_000 == 0:
                    self.training_report(rewards_data,
                                         loss_data,
                                         success_data_rand,
                                         episode_len_rand,
                                         success_data_exp,
                                         episode_len_exp,
                                         training_episode_lengths,
                                         save_path)
            s = sp.copy()
            if done:
                # s = self.env.reset(0)
                num_pre_moves = np.random.randint(low=0, high=40)
                num_pre_moves = 0
                # num_pre_moves = int(np.random.rand() < 0.5)
                s = self.env.reset(num_pre_moves)
                self.env.player = 1 + (int(np.random.rand() < 0.5) * -2)
                rewards_data.append(episode_rewards)
                episode_rewards = 0
                episode_count += 1
                training_episode_lengths.append(episode_len)
                episode_len = num_pre_moves

        best_network.save(os.path.join(save_path, 'best_network_weights'))
        return self.training_report(rewards_data,
                                    loss_data,
                                    success_data_rand,
                                    episode_len_rand,
                                    success_data_exp,
                                    episode_len_exp,
                                    training_episode_lengths,
                                    save_path)

    def optimize(self) -> float:
        '''Optimize Q-network by minimizing td-error on mini-batch sampled
        from replay buffer
        '''
        s, a, r, sp, d = self.buffer.sample(self.batch_size)
        #
        # s = torch.tensor(s, dtype=torch.float32).to(self.device).unsqueeze(0).permute(1, 0, 2, 3)
        # a = torch.tensor(a, dtype=torch.long).to(self.device)
        # r = torch.tensor(r, dtype=torch.float32).to(self.device)
        # sp = torch.tensor(sp, dtype=torch.float32).to(self.device).unsqueeze(0).permute(1, 0, 2, 3)
        # d = torch.tensor(d, dtype=torch.float32).to(self.device)

        s = torch.tensor(s, dtype=torch.float32).to(self.device)
        a = torch.tensor(a, dtype=torch.long).to(self.device)
        r = torch.tensor(r, dtype=torch.float32).to(self.device)
        sp = torch.tensor(sp, dtype=torch.float32).to(self.device)
        d = torch.tensor(d, dtype=torch.float32).to(self.device)

        q_pred = self.network(s).gather(1, a.unsqueeze(1)).squeeze()
        #
        with torch.no_grad():
            # TODO how to use done
            q_target = r + self.gamma * torch.max(self.target_network(sp), dim=1)[0] # TODO

        # if self.update_method == 'standard':
        #     with torch.no_grad():
        #         q_map_next = self.target_network(sp)
        #         q_next = torch.max(torch.flatten(q_map_next, 1), dim=1)[0]
        #         q_target = r + self.gamma * q_next * (1 - d)
        #
        # elif self.update_method == 'double':
        #     with torch.no_grad():
        #         q_map_curr = self.network(sp)
        #         # best_act = argmax2d(q_map_curr)[0]
        #         best_act = torch.argmax(q_map_curr.flatten())
        #         q_targ = self.target_network(sp).flatten()[best_act]
        #         # print(best_act)
        #         # q_targ = self.target_network(sp)[best_act]
        #         q_target = r + self.gamma * q_targ * (1 - d)

        self.optim.zero_grad()

        # assert q_pred.shape == q_target.shape
        loss = self.network.compute_loss(q_pred, q_target)
        loss.backward()

        # it is common to clip gradient to prevent instability
        # TODO
        nn.utils.clip_grad_norm_(self.network.parameters(), 10)
        self.optim.step()
        # self.scheduler.step()
        return loss.item()

    def _select_action(self, state: np.ndarray, epsilon: float = 0.) -> int:
        # if epsilon == 0.2:
        #     if np.random.random() < epsilon:
        #         return np.random.choice(self.env.legal_actions())
        #     else:
        #         return self.policy(self.env.get_observation())

        '''Performs e-greedy action selection'''
        if np.random.random() < epsilon:
            return np.random.choice(self.env.legal_actions())
            # return expert_action(self.env.board, self.env.player, np.random.choice(self.env.legal_actions()))
        # elif np.random.random() < epsilon:
        #     return expert_action(self.env.board, self.env.player, np.random.choice(self.env.legal_actions()))
        else:
            return self.policy(self.env.get_observation())

    def _select_opp_action(self, epsilon: float = 0.) -> int:
        # if epsilon == 0.2:
        #     if np.random.random() < epsilon:
        #         return np.random.choice(self.env.legal_actions())
        #     else:
        #         return self.opponent.act(self.env)

        '''Performs e-greedy action selection'''
        if np.random.random() < epsilon:
            return np.random.choice(self.env.legal_actions())
            # return expert_action(self.env.board, self.env.player, np.random.choice(self.env.legal_actions()))
        # elif np.random.random() < epsilon:
        #     return expert_action(self.env.board, self.env.player, np.random.choice(self.env.legal_actions()))
        else:
            return self.opponent.act(self.env)

    def select_action(self, env):
        return self.policy(env.get_observation())

    def compute_epsilon(self, fraction: float) -> float:
        '''Compute epsilon value based on fraction of training steps'''
        fraction = np.clip(fraction, 0., 1.)
        return (1 - fraction) * self.initial_epsilon + fraction * self.final_epsilon

    def hard_target_update(self):
        '''Copy weights of q-network to target q-network'''
        self.target_network.load_state_dict(self.network.state_dict())

    def policy(self, obs: np.ndarray) -> int:
        '''Calculates argmax of Q-function at given state'''
        t_obs = torch.tensor(obs, dtype=torch.float32,
                             device=self.device).unsqueeze(0)
        return self.network.predict(t_obs, self.env)

    def training_report(self,
                        rewards_data,
                        loss_data,
                        success_data_rand,
                        episode_len_rand,
                        success_data_exp,
                        episode_len_exp,
                        training_episode_lengths,
                        save_path):

        f, axs = plt.subplots(5, 1, figsize=(5, 10))

        axs[0].plot(np.convolve(rewards_data, np.ones(self.plotting_smoothing) / self.plotting_smoothing, 'valid'))
        axs[0].set_xlabel('episodes')
        axs[0].set_ylabel('sum of rewards')

        axs[1].plot(np.convolve(loss_data, np.ones(self.plotting_smoothing) / self.plotting_smoothing, 'valid'))
        axs[1].set_xlabel('opt steps')
        axs[1].set_ylabel('td loss')

        axs[2].plot(
            np.convolve(training_episode_lengths, np.ones(self.plotting_smoothing) / self.plotting_smoothing, 'valid'))
        axs[2].set_xlabel('opt steps')
        axs[2].set_ylabel('training episode length')

        # axs[3].plot(episode_len_simple, label='simple',color='red')
        axs[3].plot(episode_len_rand, label='random', color='green')
        axs[3].plot(episode_len_exp, label='expert', color='orange')
        axs[3].set_xlabel('trials')
        axs[3].set_ylabel('episode length')
        axs[3].legend()

        # axs[4].plot(success_data_simple, label='simple', color='red')
        axs[4].plot(success_data_rand, label='random', color='green')
        axs[4].plot(success_data_exp, label='expert', color='orange')
        axs[4].set_xlabel('trials')
        axs[4].set_ylabel('success rate')
        axs[4].legend()

        plt.tight_layout()
        plt.savefig(os.path.join(save_path, datetime.now().strftime("%Y%m%d-%H%M%S") + '-fig.png'))

        plt.close('all')

    def get_name(self):
        return "DQN Agent"

    def save(self, path):
        self.target_network.save(path)

    def load(self, path):
        self.network.load(path)

    __call__ = select_action


def main():
    env = Connect4Env()

    agent = DQNAgent(env,
                     gamma=0.9999,
                     learning_rate=1e-5,
                     buffer_size=25_000,
                     initial_epsilon=0.95,
                     final_epsilon=0.2,
                     exploration_fraction=1.,
                     target_network_update_freq=25_000,  # temproaly correlated epsiode?
                     batch_size=256,
                     device='cpu',
                     update_method='standard',
                     plotting_smoothing=5000,
                     )

    root = '/Users/szlota777/Desktop/Spring2022/Cs4910/connect_four/robo-connectfour/saved'

    save_path = os.path.join(root, datetime.now().strftime("%Y%m%d-%H%M%S"))

    os.mkdir(save_path)
    with open(os.path.join(save_path, 'parameters.txt'), 'w') as f:
        for attr in dir(agent):
            if type(getattr(agent, attr)) is int or type(getattr(agent, attr)) is float or type(
                    getattr(agent, attr)) is str:
                f.write("obj.%s = %r" % (attr, getattr(agent, attr)))
                f.write('\n')

    agent.train(200_000, save_path)
    agent.save(os.path.join(save_path, 'final_network_weights'))
    # larger learning rate less freuqent update

    print(f'Training Report:')
    # print(f'simplistic win_percentage : {round(won, 4)}')
    won, _ = simulate_against_random(agent)
    print(f'random win_percentage : {round(won, 4)}')
    won, _ = simulate_against_expert(agent)
    print(f'expert win_percentage : {round(won, 4)}')


if __name__ == "__main__":
    main()
