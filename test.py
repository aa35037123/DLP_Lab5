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
import torch.nn.functional as F
from collections import namedtuple
from torch.utils.tensorboard import SummaryWriter
from atari_wrappers import wrap_deepmind, make_atari

Transition = namedtuple('Transition', 
                        ('state', 'action', 'reward', 'next_state', 'done'))
class ReplayMemory(object):
    ## TODO ##
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        
    def push(self, *args):
        # if memory has enough space, then we can append element to it, it's because we use it to index operation 
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)


        


class Net(nn.Module):
    def __init__(self, num_classes=4, init_weights=True):
        super(Net, self).__init__()

        self.cnn = nn.Sequential(nn.Conv2d(4, 32, kernel_size=8, stride=4), # (84-8)/4 +1 = 20, shape: 16*20*20
                                        nn.ReLU(True),
                                        nn.Conv2d(32, 64, kernel_size=4, stride=2), # (20-4)/2 +1 = 9, shape:32*9*9
                                        nn.ReLU(True),
                                        nn.Conv2d(64, 64, kernel_size=3, stride=1), # (9-3)/1 +1 = 7, shape: 64x7x7
                                        nn.ReLU(True)
                                        )
        self.classifier = nn.Sequential(nn.Linear(7*7*64, 512),
                                        nn.ReLU(True),
                                        nn.Linear(512, num_classes)
                                        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = x.float() / 255.
        x = self.cnn(x)
        x = x.view(-1, 3136)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0.0)


class DQN:
    def __init__(self, args):
        self.env_raw = make_atari(args.env_name)
        self.env = wrap_deepmind(self.env_raw, frame_stack=False, episode_life=False, clip_rewards=False)
        self.num_actions = self.env.action_space.n
        self._behavior_net = Net(num_classes=self.num_actions).to(args.device)
        self._target_net = Net(num_classes=self.num_actions).to(args.device)
        # initialize target network
        self._target_net.load_state_dict(self._behavior_net.state_dict())
        self._target_net.eval()
        self._optimizer = torch.optim.Adam(self._behavior_net.parameters(), lr=args.lr, eps=1.5e-4)

        """Initialize replay buffer"""
        self._memory = ReplayMemory(capacity=args.capacity)
        self.state_counter = 0
        self.state_size = 5   # wrap 4 frams together, add 1 buffer frame

        ## config ##
        self.device = args.device
        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.freq = args.freq
        self.target_freq = args.target_freq

    def state_initialize(self):
        self.state_buffer = []
        obs = self.env.reset() # 1 frame is used to initialize
    
        for i in range(self.state_size):
            self.state_buffer.append(obs)
            
        return self.state_buffer[1:5]
    
    def select_action(self, state, epsilon, action_space):
        '''epsilon-greedy based on behavior network'''
        if random.random() < epsilon:
            action = action_space.sample()
        else:
            with torch.no_grad():
                # squeeze can remove all dimension that=1
                q_values = self._behavior_net(torch.from_numpy(np.array(state)).to(self.device).squeeze())
                # t.max(1) will return largest column value of each row.
                action = q_values.max(dim=1)[1]
                
        return action
    
    def append(self, state, action, reward, next_state, done):
        """Push a transition into replay buffer"""
        self.state_counter += 1
        self._memory.push(state, action, reward, next_state, done)

    def update(self, total_steps):
        if total_steps % self.freq == 0:
            self._update_behavior_network(self.gamma)
        if total_steps % self.target_freq == 0:
            self._update_target_network()

    def _update_behavior_network(self, gamma):
        # sample a minibatch of transitions
        transitions = self._memory.sample(self.batch_size)
        # shape of q_value : (batch_size x 1(action))
        batch = Transition(*zip(*transitions))
        
        state_batch = torch.tensor(np.array(batch.state,dtype=np.float32),device=self.device,dtype=torch.float32).to(self.device)
        action_batch = torch.tensor(batch.action,device=self.device).unsqueeze(1).to(self.device)
        reward_batch = torch.tensor(batch.reward,device=self.device,dtype=torch.float32).unsqueeze(1).to(self.device)
        next_s_batch = torch.tensor(np.array(batch.next_state,dtype=np.float32),device=self.device,dtype=torch.float32).to(self.device)
        done_batch = torch.tensor(np.array(batch.done,dtype=np.float32),device = self.device,dtype=torch.float32).unsqueeze(1).to(self.device)
        # squeeze remove all dimension that = 1
        q_value = self._behavior_net(state_batch.squeeze()).gather(dim=1, index=action_batch.long())
        
        with torch.no_grad():
            q_next = self._target_net(next_s_batch.squeeze()).max(dim=1)[0].detach().unsqueeze(1)
            q_target = reward_batch + gamma*q_next*(1-done_batch)

        # Compute Huber loss
        loss = F.mse_loss(q_value, q_target)
        
        self._optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self._behavior_net.parameters(), 5) # limit grad of network should not exceed 5
        self._optimizer.step()

    def _update_target_network(self):
        '''update target network by copying from behavior network'''
        self._target_net.load_state_dict(self._behavior_net.state_dict())

    def save(self, model_path, checkpoint=False):
        if checkpoint:
            torch.save({
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
def dim(a):
    if not type(a) == list:
        return []
    return [len(a)] + dim(a[0])
def get_state(obs):
    state = np.array(obs)
    state = state.transpose((2, 0, 1))
    state = state.astype(np.float32)
    state = torch.from_numpy(state)
    return state.unsqueeze(0) # add 1 dimension to state

def train(args, agent, writer):
    print('Start Training')
    action_space = agent.env.action_space
    total_steps, epsilon = 0, args.epsilon
    done = True
    ewma_reward = 0.0   
    total_reward = 0.0
    for step in range(1, args.num_step+1):
        if done :
            state = agent.state_initialize()
            obs, reward, done, _ = agent.env.step(1) # Fire first
            if step != 1:
                ewma_reward = 0.05 * total_reward + (1 - 0.05) * ewma_reward
                writer.add_scalar('Train/Episode Reward', total_reward, step)
                writer.add_scalar('Train/Ewma Reward', ewma_reward, step)
                print('Step: {}\tTotal reward: {:.2f}\tEwma reward: {:.2f}\tEpsilon: {:.3f}'
                        .format(total_steps, total_reward, ewma_reward, epsilon))
                total_reward = 0.0
            
        if total_steps < args.warmup:
                action = action_space.sample()
        else:
            action = agent.select_action(state, epsilon, action_space) # shape of state : (4x84x84)
            # decay epsilon
            epsilon -= (1 - args.eps_min) / args.eps_decay
            epsilon = max(epsilon, args.eps_min)
            
        obs, reward, done, _ = agent.env.step(action)
        agent.state_buffer.pop(0)
        agent.state_buffer.append(obs)
        next_state = agent.state_buffer[1:5]
        agent.append(state, action, reward, next_state, done)
            
        total_reward += reward
        total_steps += 1
        state = next_state
        
        if total_steps >= args.warmup:
                agent.update(total_steps)
            
        if total_steps % args.eval_freq == 0:
            """You can write another evaluate function, or just call the test function."""
            test(args, agent, writer)
            agent.save(args.model + "dqn_" + str(total_steps) + ".pt", checkpoint=True)
        
        
    # for episode in range(args.episode):
    #     total_reward = 0.0
    #     state = agent.state_initialize()
    #     obs, reward, done, _ = agent.env.step(1) # fire first !!!
        
    #     for t in itertools.count(start=1):
    #         # agent.env.render()
    #         if total_steps < args.warmup:
    #             action = action_space.sample()
    #         else:
    #             # select action
    #             # print(f'ep={episode}, t={t}')
    #             # print(f'State : {state}')
    #             action = agent.select_action(state, epsilon, action_space) # shape of state : (4x84x84)
    #             # decay epsilon
    #             epsilon -= (1 - args.eps_min) / args.eps_decay
    #             epsilon = max(epsilon, args.eps_min)
    #         # execute action
    #         obs, reward, done, _ = agent.env.step(action)
    #         # done means 5 lives are gone, so there are no next step(state can not be none) 
    #         agent.state_buffer.pop(0)
    #         agent.state_buffer.append(obs)
    #         next_state = agent.state_buffer[1:5]
            
    #         # reward = torch.tensor([reward], device=agent.device)
    #         agent.append(state, action, reward, next_state, done)
            
    #         total_reward += reward
    #         total_steps += 1
    #         state = next_state
            
    #         if total_steps >= args.warmup:
    #             agent.update(total_steps)
            
    #         if total_steps % args.eval_freq == 0:
    #             """You can write another evaluate function, or just call the test function."""
    #             test(args, agent, writer)
    #             agent.save(args.model + "dqn_" + str(total_steps) + ".pt")

            
    #         if done:
    #             ewma_reward = 0.05 * total_reward + (1 - 0.05) * ewma_reward
    #             writer.add_scalar('Train/Episode Reward', total_reward, episode)
    #             writer.add_scalar('Train/Ewma Reward', ewma_reward, episode)
    #             print('Step: {}\tEpisode: {}\tLength: {:3d}\tTotal reward: {:.2f}\tEwma reward: {:.2f}\tEpsilon: {:.3f}'
    #                     .format(total_steps, episode, t, total_reward.item(), ewma_reward.item(), epsilon))
    #             break
    agent.env.close()


def test(args, agent, writer):
    print('Start Testing')
    action_space = agent.env.action_space
    e_rewards = []
    
    for i in range(args.test_episode):
        state = agent.state_initialize()
        e_reward = 0
        done = False

        while not done:
            action = agent.select_action(state, args.test_epsilon, action_space)
            obs, reward, done, _ = agent.env.step(action)
            e_reward += reward
            agent.state_buffer.pop(0)
            agent.state_buffer.append(obs)
            state = agent.state_buffer[1:5]
            
        writer.add_scalar('Test/Episode Reward', e_reward, i)
        print('episode {}: {:.2f}'.format(i+1, e_reward))
        e_rewards.append(e_reward)

    agent.env.close()
    print('Average Reward: {:.2f}'.format(float(sum(e_rewards)) / float(args.test_episode)))


def main():
    ## arguments ##
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--env_name', type=str, default="BreakoutNoFrameskip-v4",
                            help='name of Mujoco environement')
    parser.add_argument('-d', '--device', default='cuda')
    parser.add_argument('-m', '--model', default='ckpt/')
    parser.add_argument('--logdir', default='log/atari')
    # train
    parser.add_argument('--warmup', default=50000, type=int)
    parser.add_argument('--episode', default=20000, type=int)
    parser.add_argument('--capacity', default=1000000, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--lr', default=0.0000625, type=float)
    parser.add_argument('--num_step', default=50000000, type=int)
    parser.add_argument('--epsilon', default=1., type=float)
    parser.add_argument('--eps_decay', default=1000000, type=float)
    parser.add_argument('--eps_min', default=0.1, type=float)
    parser.add_argument('--gamma', default=.99, type=float)
    parser.add_argument('--freq', default=4, type=int)
    parser.add_argument('--target_freq', default=10000, type=int)
    parser.add_argument('--eval_freq', default=200000, type=int)
    # test
    parser.add_argument('--test_only', action='store_true')
    parser.add_argument('-tmp', '--test_model_path', default='ckpt/dqn_1800000.pt')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--test_episode', default=10, type=int)
    parser.add_argument('--seed', default=20230422, type=int)
    parser.add_argument('--test_epsilon', default=0.01, type=float)
    args = parser.parse_args()

    ## main ##
    agent = DQN(args)
    writer = SummaryWriter(args.logdir)
    train_model_path = './ckpt/dqn_1800000.pt'
    test_model_path = './ckpt/dqn_9000000.pt'
    if args.test_only:
        agent.load(test_model_path)
        test(args, agent, writer)
    else:
        # agent.load(train_model_path)
        train(args, agent, writer)
        


if __name__ == '__main__':
    main()
