'''DLP DQN Lab'''
__author__ = 'chengscott'
__copyright__ = 'Copyright 2020, NCTU CGI Lab'
import argparse
from collections import deque
import itertools
import random
import time

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter



class ReplayMemory:
    __slots__ = ['buffer']

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, *transition):
        # (state, action, reward, next_state, done)
        self.buffer.append(tuple(map(tuple, transition))) # use map to transform element of transition into tuple

    def sample(self, batch_size, device):
        '''sample a batch of transition tensors'''
        transitions = random.sample(self.buffer, batch_size) # choose {batch_size} samples from buffer
        return (torch.tensor(x, dtype=torch.float, device=device)
                for x in zip(*transitions))


class Net(nn.Module):
    def __init__(self, state_dim=8, action_dim=4, hidden_dim=(400, 300)):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim[0])
        self.fc2 = nn.Linear(hidden_dim[0], hidden_dim[1])
        self.fc3 = nn.Linear(hidden_dim[1], action_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.fc1(x))
        out = self.relu(self.fc2(out))
        out = self.fc3(out)
        return out

class DQN:
    def __init__(self, args):
        # behavior net choose action with given state 
        self._behavior_net = Net().to(args.device)
        self._target_net = Net().to(args.device)
        # initialize target network
        # target network evaluate target Q value, it can not choose action directly  
        # target network offer a steady Q value 
        self._target_net.load_state_dict(self._behavior_net.state_dict()) 
        # first param of Adam means the params that need to 
        self._optimizer = optim.Adam(self._behavior_net.parameters(), lr=args.lr)
        # memory
        self._memory = ReplayMemory(capacity=args.capacity)

        ## config ##
        self.device = args.device
        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.freq = args.freq # update behavior network every self.freq iteration
        self.target_freq = args.target_freq # update target network every self.target_freq iteration

    def select_action(self, state, epsilon, action_space):
        '''epsilon-greedy based on behavior network'''
        # we have epsilon prob to randomly choose action
        if random.random() < epsilon: # expore
            return action_space.sample()
        else: # exploit 
            with torch.no_grad():
                # view(batch_size, n), if n=-1, output=[batch_size, product of all dimension]
                # q_values is (batch_size x action)
                q_values = self._behavior_net(torch.from_numpy(state).view(1, -1).to(self.device))
                # print(f'Shape of q_values : {q_values}')
                # dim=0 : compare between channel, dim=1 : compare between row, dim=2 : compare between col
                # max: return [max value, max idx]
                actions = q_values.max(dim=1)[1].item()
        return actions
    def append(self, state, action, reward, next_state, done):
        self._memory.append(state, [action], [reward / 10], next_state, # reduce value of reward, avoid gradient explosion
                            [int(done)])

    def update(self, total_steps):
        # print(self._memory.sample(
        #     self.batch_size, self.device))
        if total_steps % self.freq == 0:
            self._update_behavior_network(self.gamma)
        if total_steps % self.target_freq == 0:
            self._update_target_network()

    def _update_behavior_network(self, gamma):
        # sample a minibatch of transitions
        samples = self._memory.sample(
            self.batch_size, self.device)
        # each data in samples will show up in order of (state, action, reward, next_state, done)
        # for data in samples:
        #     print(f'Sample : {data}')
        (state, action, reward, next_state, done) = self._memory.sample(
            self.batch_size, self.device)
        # _behavior_net returns batch_size x num_actions
        q_value = self._behavior_net(state).gather(dim=1, index=action.long())
        with torch.no_grad():
            # shape of q_next is (batch_size x 1)
            q_next = self._target_net(next_state).max(dim=1)[0].view(-1, 1) # -1 means flatten the tensor, and the last dimensio=1
            q_target = reward + gamma*q_next*(1-done)
        criterion = nn.MSELoss() # loss of DQN is base on MSE
        loss = criterion(q_value, q_target)
        
        # update network, back propagation, optimize
        self._optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self._behavior_net.parameters(), 5)
        self._optimizer.step()

    def _update_target_network(self):
        '''update target network by copying from behavior network'''
        self._target_net.load_state_dict(self._behavior_net.state_dict())

    def save(self, model_path, checkpoint=False):
        if checkpoint:
            torch.save(
                {
                    'behavior_net': self._behavior_net.state_dict(),
                    'target_net': self._target_net.state_dict(),
                    'optimizer': self._optimizer.state_dict(),
                }, model_path)
        else:
            torch.save({
                'behavior_net': self._behavior_net.state_dict(),
            }, model_path)

    def load(self, model_path, checkpoint=False):
        model = torch.load(model_path)
        self._behavior_net.load_state_dict(model['behavior_net'])
        if checkpoint:
            self._target_net.load_state_dict(model['target_net'])
            self._optimizer.load_state_dict(model['optimizer'])


def train(args, env, agent, writer):
    print('Start Training')
    action_space = env.action_space
    # 1 step means 1 
    total_steps, epsilon = 0, args.epsilon
    ewma_reward = 0
    for episode in range(args.episode):
        total_reward = 0
        state = env.reset() # reset env into intial status 
        # in every epoch, there is 1 simulation, terminates when condition reached
        epsilon = max(epsilon * args.eps_decay, args.eps_min)
        for t in itertools.count(start=1): # use for time counting, default step=1
            if t == 1:
                state = state[0]

            # select action 
            if total_steps < args.warmup: # adopt random action 
                action = action_space.sample()
            else: # choose action based on epilson greedy
                action = agent.select_action(state, epsilon, action_space)
            # execute action
            next_state, reward, done, truncated, info = env.step(action)
            # store transition
            # print(f'state : {state}, action : {action}, reward : {reward}')
            # print(f'next_state : {next_state}, done : {done}')
            # print(f't:{t}, State : {state}')
            # state = torch.tensor(state, dtype=torch.float)
            # state = tuple(state[0])
            # float_state = (float(x) for x in state)
            # print(f't:{t}, next_state : {next_state}')
            # next_state = tuple(next_state[0])
            # float_next_state = (float(x) for x in next_state[0])
            # agent.append(float_state, float(action), float(reward), float_next_state, float(done))
            agent.append(state, action, reward, next_state, done)
            if total_steps >= args.warmup:
                agent.update(total_steps)

            state = next_state
            total_reward += reward
            total_steps += 1
            if done:
                # EWMA : Exponential Weighted Moving Average Reward
                # give history reward more weight, reduce the oscillate of reward curve
                ewma_reward = 0.05 * total_reward + (1 - 0.05) * ewma_reward 
                writer.add_scalar('Train/Episode Reward', total_reward, total_steps)
                writer.add_scalar('Train/Ewma Reward', ewma_reward, total_steps)
                print(
                    'Step: {}\tEpisode: {}\tLength: {:3d}\tTotal reward: {:.2f}\tEwma reward: {:.2f}\tEpsilon: {:.3f}'
                    .format(total_steps, episode, t, total_reward, ewma_reward,
                            epsilon))
                break
    env.close()


def test(args, env, agent, writer):
    print('Start Testing')
    action_space = env.action_space
    epsilon = args.test_epsilon
    seeds = (args.seed + i for i in range(10)) # use random seed to ensure the variaty of env
    rewards = []
    # test for each seed
    for n_episode, seed in enumerate(seeds):
        total_reward = 0
        # env.seed(seed)
        state = env.reset(seed=seed)
        for t in itertools.count(start=1): # start 1 epoch
            # env.render() # show the progress of play
            print(f'episode={n_episode}, t : {t}')
            if t == 1:
                state = state[0]
            action = agent.select_action(state, epsilon, action_space)
            next_state, reward, done, _, _ = env.step(action)
            
            state = next_state
            total_reward += reward
            if done:
                writer.add_scalar('Test/Episode Reward', total_reward, n_episode)
                print(f'Step : {t}, total reward : {total_reward}')
                rewards.append(total_reward)
                break
            
    print('Average Reward', np.mean(rewards))
    env.close()


def main():
    ## arguments ##
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-d', '--device', default='cuda')
    parser.add_argument('-m', '--model', default='dqn.pth')
    parser.add_argument('--logdir', default='log/dqn')
    # train
    parser.add_argument('--warmup', default=10000, type=int)
    parser.add_argument('--episode', default=1200, type=int)
    parser.add_argument('--capacity', default=10000, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=.0005, type=float)
    parser.add_argument('--epsilon', default=1., type=float)
    parser.add_argument('--eps_decay', default=.995, type=float)
    parser.add_argument('--eps_min', default=.01, type=float)
    parser.add_argument('--gamma', default=.99, type=float)
    parser.add_argument('--freq', default=4, type=int)
    parser.add_argument('--target_freq', default=100, type=int)
    # test
    parser.add_argument('--test_only', action='store_true')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--seed', default=20200519, type=int)
    parser.add_argument('--test_epsilon', default=.001, type=float)
    args = parser.parse_args()

    ## main ##
    env = gym.make('LunarLander-v2')
    agent = DQN(args)
    writer = SummaryWriter(args.logdir)
    # read_model_path = 'dqn_ep=300.pth'
    write_model_path = f'dqn_ep={args.episode}.pth'
    if not args.test_only:
        # agent.load(model_path=read_model_path, checkpoint=True)
        train(args, env, agent, writer)
        agent.save(write_model_path, checkpoint=True)
    agent.load(model_path=write_model_path)
    test(args, env, agent, writer)


if __name__ == '__main__':
    main()
