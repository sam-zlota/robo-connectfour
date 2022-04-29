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

from agent import expert_action

from evaluation import SuccessTracker



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
        self.network = QNetwork().to(device)
        self.target_network = QNetwork().to(device)
        self.hard_target_update()

        self.optim = torch.optim.Adam(self.network.parameters(),
                                      lr=learning_rate)

        self.opponent = AgentOpponent(self.device, self.env.action_space.n)
        self.opponent.update(self.target_network)

        self.update_method = update_method
        self.plotting_smoothing = plotting_smoothing
        # np.random.seed(seed)
        # torch.manual_seed(seed)
        # if device == 'cuda':
        #     torch.cuda.manual_seed(seed)

    def train(self, num_steps: int, save_path) -> None:
        '''Trains q-network for given number of environment steps, plots
        rewards and loss curve
        '''

        training_metrics = {
            'rewards_data': [],
            'loss_data': [],
            'training_episode_lengths': []
        }
        success_tracker = SuccessTracker()

        episode_count = 0
        episode_rewards = 0
        opt_count = 0
        s = self.env.reset()
        # play_history = np.zeros((num_steps, 50, 3), dtype=np.int8)

        pbar = tqdm(range(1, num_steps + 1))
        episode_len = 0
        episode_number = 0

        for step in pbar:
            epsilon = self.compute_epsilon(step / (self.exploration_fraction * num_steps))

            a = self._select_action(s, epsilon)
            sp, r, done1, _ = self.env.step(a)
            # play_history[episode_number][episode_len] = (self.env.player * -1, a, r)
            episode_len += 1
            next_state = sp.copy()
            # if not done1:
            #     a_opp = self._select_opp_action(epsilon)
            #     sp2, r_opp, done2, _ = self.env.step(a_opp)
            #     play_history[episode_number][episode_len] = (self.env.player * -1, a_opp, r_opp)
            #     if r_opp == 1:
            #         r = -1
            #         play_history[episode_number][episode_len - 1][2] = -1
            #     next_state = sp2.copy()
            #     episode_len += 1
            #     self.buffer.add_transition(s=sp, a=a_opp, r=r_opp, sp=sp2, d=done2)

            episode_rewards += r

            self.buffer.add_transition(s=s, a=a, r=r, sp=sp, d=done1)

            s = next_state

            # optimize
            if self.buffer.length > self.batch_size:
                loss = self.optimize()
                opt_count += 1
                training_metrics['loss_data'].append(loss)

                if opt_count % self.target_network_update_freq == 0:
                    self.hard_target_update()

            # if step % 5000 == 0:
            #     self.opponent.update(self.network)

            if step % 200_000 == 0:
                self.save(os.path.join(save_path, datetime.now().strftime("%Y%m%d-%H%M%S") + '-weights'))

            if step % 10_000 == 0:
                success_tracker.evaluate(self)
                if step % 50_000 == 0:
                    success_tracker.plot_metrics(save_path)
                    self.training_report(training_metrics, save_path)

            # if done1 or done2:
            if done1:
                #num_pre_moves = np.random.randint(low=0, high=35)
                num_pre_moves = 0
                s = self.env.reset(num_pre_moves)
                # random starting player
                # self.env.player = 1 + (int(np.random.rand() < 0.5) * -2)
                training_metrics['rewards_data'].append(episode_rewards)
                episode_rewards = 0
                episode_count += 1
                training_metrics['training_episode_lengths'].append(episode_len)
                episode_len = num_pre_moves

                episode_number += 1


    def optimize(self) -> float:
        '''Optimize Q-network by minimizing td-error on mini-batch sampled
        from replay buffer
        '''
        s, a, r, sp, d = self.buffer.sample(self.batch_size)

        s = torch.tensor(s, dtype=torch.float32).to(self.device)
        a = torch.tensor(a, dtype=torch.long).to(self.device)
        r = torch.tensor(r, dtype=torch.float32).to(self.device)
        sp = torch.tensor(sp, dtype=torch.float32).to(self.device)
        d = torch.tensor(d, dtype=torch.float32).to(self.device)

        q_pred = self.network(s).gather(1, a.unsqueeze(1)).squeeze() # TODO?

        with torch.no_grad():
            q_target = r + self.gamma * torch.max(self.target_network(sp), dim=1)[0] * (1-d)  # TODO should this be multipled by (1-d)

        self.optim.zero_grad()

        loss = self.network.compute_loss(q_pred, q_target)
        loss.backward()

        # it is common to clip gradient to prevent instability
        nn.utils.clip_grad_norm_(self.network.parameters(), 10)
        self.optim.step()
        return loss.item()

    def _select_action(self, state: np.ndarray, epsilon: float = 0.) -> int:
        '''Performs e-greedy action selection'''
        if np.random.random() < epsilon:
            if epsilon < 0.4:
                return np.random.choice(self.env.legal_actions())
            else:
            #uncomment for demonstration
                return expert_action(self.env.board, self.env.player, np.random.choice(self.env.legal_actions()))
        else:
            return self.policy(self.env.get_observation(), self.env)

    # def _select_opp_action(self, epsilon: float = 0.) -> int:
    #     '''Performs e-greedy action selection'''
    #     if np.random.random() < epsilon:
    #         return np.random.choice(self.env.legal_actions())
    #         # uncomment for demonstration
    #         # return expert_action(self.env.board, self.env.player, np.random.choice(self.env.legal_actions()))
    #     else:
    #         # choose random of recent past selfs
    #         return self.opponent.act(self.env)

    def select_action(self, env):
        return self.policy(env.get_observation(), env)

    def compute_epsilon(self, fraction: float) -> float:
        '''Compute epsilon value based on fraction of training steps'''
        fraction = np.clip(fraction, 0., 1.)
        return (1 - fraction) * self.initial_epsilon + fraction * self.final_epsilon

    def hard_target_update(self):
        '''Copy weights of q-network to target q-network'''
        self.target_network.load_state_dict(self.network.state_dict())

    def policy(self, obs: np.ndarray, env) -> int:
        '''Calculates argmax of Q-function at given state'''
        winning_moves = env.get_winning_moves()
        if winning_moves:
            return np.random.choice(winning_moves)
        t_obs = torch.tensor(obs, dtype=torch.float32,
                             device=self.device).unsqueeze(0)
        return self.network.predict(t_obs, env)

    def training_report(self,
                        training_metrics, save_path):

        f, axs = plt.subplots(3, 1, figsize=(10, 18))

        axs[0].plot(
            np.convolve(training_metrics['rewards_data'], np.ones(self.plotting_smoothing) / self.plotting_smoothing,
                        'valid'))
        axs[0].set_xlabel('episodes')
        axs[0].set_ylabel('sum of rewards')
        axs[0].set_title('Sum of Rewards')

        axs[1].plot(
            np.convolve(training_metrics['loss_data'], np.ones(self.plotting_smoothing) / self.plotting_smoothing,
                        'valid'))
        axs[1].set_xlabel('opt steps')
        axs[1].set_ylabel('td loss')
        axs[1].set_title('Loss')

        axs[2].plot(
            np.convolve(training_metrics['training_episode_lengths'],
                        np.ones(self.plotting_smoothing) / self.plotting_smoothing, 'valid'))
        axs[2].set_xlabel('opt steps')
        axs[2].set_ylabel('training episode length')
        axs[2].set_title('Training Episode Lengths')

        plt.tight_layout()
        plt.savefig(os.path.join(save_path, datetime.now().strftime("%Y%m%d-%H%M%S") + '-training-fig.png'))

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
                     gamma=0.99,
                     learning_rate=1e-5,
                     buffer_size=150_000,
                     initial_epsilon=0.95,
                     final_epsilon=0.01,
                     exploration_fraction=1.,
                     target_network_update_freq=40_000,
                     batch_size=256,
                     device='cpu',
                     update_method='standard',
                     plotting_smoothing=5000,
                     )

    # where u want run info to be saved
    # will save a parameters.txt file
    # best_network_weights and final_network_weights_file
    # as well as a plot after each 20_000 steps
    root = '/Users/szlota777/Desktop/Spring2022/Cs4910/connect_four/robo-connectfour/saved'

    save_path = os.path.join(root, datetime.now().strftime("%Y%m%d-%H%M%S"))

    os.mkdir(save_path)
    with open(os.path.join(save_path, 'parameters.txt'), 'w') as f:
        for attr in dir(agent):
            if type(getattr(agent, attr)) is int or type(getattr(agent, attr)) is float or type(
                    getattr(agent, attr)) is str:
                f.write("obj.%s = %r" % (attr, getattr(agent, attr)))
                f.write('\n')


    # path_to_existing_agent = '/Users/szlota777/Desktop/Spring2022/CS4910/connect_four/robo-connectfour/saved/20220420-194327/final_network_weights'
    # agent.load(path_to_existing_agent)
    agent.train(5_000_000, save_path)
    agent.save(os.path.join(save_path, 'final_network_weights'))


if __name__ == "__main__":
    main()
