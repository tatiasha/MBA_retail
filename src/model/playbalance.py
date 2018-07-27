from supermarket_model import Market

env = Market()
#env.reset()

for i in range(10):
    rew = env.step(i)
    print("new state = {}".format(env.state))
    print("reward = {}".format(rew))

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
