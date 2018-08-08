# -*- coding: utf-8 -*-
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from supermarket_model import Market
from torch.autograd import Variable

env = Market()

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):

    def __init__(self):
        super(DQN, self).__init__()
        self.l1 = nn.Linear(51, 512)
        self.l2 = nn.Linear(512, 51)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return x


BATCH_SIZE = 200
GAMMA = 0.9
TARGET_UPDATE = 10

policy_net = DQN()
target_net = DQN()
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(500)

def plot_durations(ave):
    plt.clf()
    rewards = torch.tensor(rew, dtype=torch.float)
    plt.plot(rewards.numpy(), label='reward')
    plt.title('Training...')
    plt.xlabel('Client')
    plt.ylabel('Reward')
    plt.plot(ave, label="average reward")
    plt.legend()

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))
    state_batch = batch.state
    next_state_batch = torch.tensor(batch.next_state, dtype=torch.float32)
    reward_batch = torch.tensor(batch.reward, dtype=torch.float32)  # torch.cat(batch.reward)
    state_batch = torch.tensor(state_batch, dtype = torch.float32)
    state_action_values = policy_net(state_batch)
    next_state_values = target_net(next_state_batch).max(1)[0].detach()
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    state_action_values = state_action_values.max(1)[0].detach()
    state_action_values = Variable(state_action_values.data, requires_grad=True)
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        if param.grad is not None:
            param.grad.data.clamp_(-1, 1)
    optimizer.step()


num_episodes = 50000
env.reset()
rew = []
average = []
n = np.random.randint(0, 100)
state = env.get_vector_state(n)
for i_episode in range(num_episodes):
    reward = env.step(n)  # do action
    action = env.action  # which action
    rew.append(reward)
    average.append(np.mean(rew))
    reward = torch.tensor([reward])
    n = np.random.randint(0, 100)
    next_state = env.get_vector_state(n)  # new state
    print(i_episode)
    memory.push(state, action, next_state, reward)
    state = next_state
    optimize_model()
    plot_durations(average)
    # Update the target network
if i_episode % TARGET_UPDATE == 0:
    target_net.load_state_dict(policy_net.state_dict())

print('Complete')

plt.show()
