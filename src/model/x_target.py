import pandas as pd


# -*- coding: utf-8 -*-
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from model_v3 import Market
from model_v3 import Client

from torch.autograd import Variable
N_components = 51
agent = Market()
env = Client(agent, N_components)

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
        self.l1 = nn.Linear(N_components, 128)
        self.l2 = nn.Linear(128, N_components)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))

        return x


BATCH_SIZE = 128
GAMMA = 0.99
TARGET_UPDATE = 10
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
steps_done = 0

policy_net = DQN()
target_net = DQN()
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(1000)

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            act = torch.tensor(state, dtype=torch.float32)
            act = policy_net(act)
            return act.numpy()
    else:
        if N_components < 51:
            q = [np.random.uniform(-1, 1) for i in range(N_components)]
        else:
            q = np.random.randint(2, size=51)
        return q

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))
    state_batch = batch.state
    next_state_batch = torch.tensor(batch.next_state, dtype=torch.float32)
    reward_batch = torch.tensor(batch.reward, dtype=torch.float32)  # torch.cat(batch.reward)
    state_batch = torch.tensor(state_batch, dtype=torch.float32)
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


num_episodes = 6000
env.reset()
rew = []
average = []
average_all = []
purchasing = []
target = []
for i_episode in range(num_episodes):
    n = i_episode
    state = env.vector_states[n]
    action = select_action(state)
    reward, r_next_state, p_rec = env.step(n, action)
    purchasing.append(state)
    target.append(p_rec)
    print(i_episode)
    next_state = r_next_state + state
    reward = torch.tensor([reward])

result = pd.DataFrame()
result['Purchasing'] = purchasing
result["Target"] = target
result.to_csv("E:\Projects\MBA_retail\\x_target_3.csv")
