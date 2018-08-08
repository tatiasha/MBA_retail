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
from model_v2 import Market
from model_v2 import Client


from torch.autograd import Variable

agent = Market()
env = Client(agent)

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
        self.l1 = nn.Linear(51, 128)
        self.l2 = nn.Linear(128, 51)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return x


BATCH_SIZE = 100
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
memory = ReplayMemory(200)

def plot_durations(ave):
    plt.figure(2)
    plt.clf()
    rewards = torch.tensor(rew, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(rewards.numpy(),label='reward')
    # Take 100 episode averages and plot them too
    plt.plot(ave, label="average reward")

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())
    #plt.savefig('E:\\result.png')

    # plt.clf()
    # rewards = torch.tensor(rew, dtype=torch.float)
    # plt.plot(rewards.numpy(), label='reward')
    # plt.title('Training...')
    # plt.xlabel('Client')
    # plt.ylabel('Reward')
    # plt.plot(ave, label="average reward")
    # plt.legend()
    # plt.pause(0.001)  # pause a bit so that plots are updated
    # if is_ipython:
    #     display.clear_output(wait=True)
    #     display.display(plt.gcf())
    # plt.savefig('E:\\result.png')


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
        q = np.random.randint(2, size=51)
        return q
        #return torch.tensor([[random.randrange(1)]], dtype=torch.long)

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
n = np.random.randint(0, 500)
state = env.vector_states[n]
for i_episode in range(num_episodes):
    # kak doljno byt'
    action = select_action(state)
    reward = env.step(n, action)

    # reward = env.step(n)  # do action
    # r_c = reward
    # action = env.action  # which action
    rew.append(reward)
    average.append(np.mean(rew[-10:]))
    reward = torch.tensor([reward])
    n = np.random.randint(0, 500)
    next_state = env.vector_states[n]  # new state
    print(i_episode, np.mean(rew))
    memory.push(state, action, next_state, reward)
    state = next_state
    optimize_model()
    plot_durations(average)
    # Update the target network
if i_episode % TARGET_UPDATE == 0:
    target_net.load_state_dict(policy_net.state_dict())

print('Complete')

plt.show()
