import gym
from gym import envs, wrappers

def getAllEnvName():
    allEnvName = list("")
    for elist in envs.registry.all():
        allEnvName.append(elist.id)
    return allEnvName

#print("\n".join(sorted(getAllEnvName())))

#env = gym.make('MountainCar-v0') # MountainCarContinuous-v0
env = gym.make('MountainCarContinuous-v0')
#env = wrappers.Monitor(env, '../Video/mountainCar-experiment-1', force = True)
print(env.action_space)
#print(env.observation_space.high)
#print(env.observation_space.low)

# defined as solving as getting average reward of 90 over 100 consecutive trials.

for i_episode in range(20):
    observation = env.reset()
    for t in range(200):
        env.render()
        #print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
