from model_v2 import Market
from model_v2 import Client

agent = Market()
env = Client(agent)
env.reset()
N = 100
for i in range(N):
    print(i)
    rew = env.step(i)
    print("new state = {}".format(env.state))
    print("reward = {}".format(rew))

print(len(env.states))

'''''''''
rew = env.step(1)
print("reward = {}".format(rew[1]))
print("new state = {}".format(env.state))
rew = env.step(1)
print("reward = {}".format(rew[1]))
print("new state = {}".format(env.state))
rew = env.step(0)
print("reward = {}".format(rew[1]))
print("new state = {}".format(env.state))
rew = env.step(0)
print("reward = {}".format(rew[1]))
print("new state = {}".format(env.state))
rew = env.step(0)
print("reward = {}".format(rew[1]))
print("new state = {}".format(env.state))
rew = env.step(1)
print("reward = {}".format(rew[1]))
print("new state = {}".format(env.state))
rew = env.step(1)
print("reward = {}".format(rew[1]))
'''