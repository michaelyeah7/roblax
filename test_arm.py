# %%
import os
import glob
import time
from datetime import datetime

import torch
import numpy as np

import gym
import gym_rbdl
# import roboschool

# import pybullet_envs

from PPO import PPO



from jbdl.rbdl.tools import plot_model
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D

#################################### Testing ###################################



# %%
print("============================================================================================")

################## hyperparameters ##################

# env_name = "CartPole-v1"
# has_continuous_action_space = False
# max_ep_len = 400
# action_std = None


# env_name = "LunarLander-v2"
# has_continuous_action_space = False
# max_ep_len = 300
# action_std = None


# env_name = "BipedalWalker-v2"
# has_continuous_action_space = True
# max_ep_len = 1500           # max timesteps in one episode
# action_std = 0.1            # set same std for action distribution which was used while saving


env_name = "jbdl_arm-v0"
has_continuous_action_space = True
max_ep_len = 1000           # max timesteps in one episode
action_std = 0.1            # set same std for action distribution which was used while saving


render = True              # render environment on screen
frame_delay = 0             # if required; add delay b/w frames


total_test_episodes = 10    # total num of testing episodes

K_epochs = 80               # update policy for K epochs
eps_clip = 0.2              # clip parameter for PPO
gamma = 0.99                # discount factor

lr_actor = 0.0003           # learning rate for actor
lr_critic = 0.001           # learning rate for critic

#####################################################


env = gym.make(env_name)
# state space dimension
state_dim = env.observation_space.shape[0]

# action space dimension
if has_continuous_action_space:
    action_dim = env.action_space.shape[0]
else:
    action_dim = env.action_space.n


# initialize a PPO agent
ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std)


# preTrained weights directory

random_seed = 0             #### set this to load a particular checkpoint trained on random seed
run_num_pretrained = 0      #### set this to load a particular checkpoint num


directory = "PPO_preTrained_forward_reward_joint_limit" + '/' + env_name + '/'
checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained)
print("loading network from : " + checkpoint_path)

ppo_agent.load(checkpoint_path)

print("--------------------------------------------------------------------------------------------")
# %%

for ep in range(1, total_test_episodes+1):
    ep_reward = 0
    state = env.reset()
    for t in range(1, max_ep_len+1):
        action = ppo_agent.select_action(state)
        state, reward, done, _ = env.step(action)
        ep_reward += reward

        if render:      
            env.osim_render()


# %%
