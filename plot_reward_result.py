import numpy as np
import pandas as pd
import math as math
import matplotlib.pyplot as plt


arm_result = pd.read_csv('SAC_arm_result.csv')
cartpole_result = pd.read_csv('SAC_cartpole_result.csv')
Pendulum_result = pd.read_csv('SAC_Pendulum_result.csv')
PPO_arm_result = pd.read_csv('arm_result.csv')
PPO_cartpole_result = pd.read_csv('cartpole_result.csv')
PPO_Pendulum_result = pd.read_csv('Pendulum_result.csv')


plt.figure()
plt.subplot(2, 3, 1)
temp = arm_result['reward'].diff()/arm_result['reward']
print(temp)
count = 0
for i in range(len(temp)):
    if np.abs(temp[i]) < 0.001:
        if count == 10:
            break
        count += 1
plt.plot(arm_result['timestep'][:i],arm_result['reward'][:i])
plt.title("SAC Arm Env timestep vs Rewards")
plt.subplot(2, 3, 2)
temp = Pendulum_result['reward'].diff()/Pendulum_result['reward']
print(temp)
count = 0
for i in range(len(temp)):
    if np.abs(temp[i]) < 0.001:
        if count == 10:
            break
        count += 1
plt.plot(Pendulum_result['timestep'][:i],Pendulum_result['reward'][:i])
plt.title("SAC Pendulum Env timestep vs Rewards")
plt.subplot(2, 3, 3)
temp = cartpole_result['reward'].diff()/cartpole_result['reward']
print(temp)
count = 0
for i in range(len(temp)):
    if np.abs(temp[i]) < 0.001:
        if count == 10:
            break
        count += 1
plt.plot(cartpole_result['timestep'][:i],cartpole_result['reward'][:i])
plt.title("SAC Cartpole Env timestep vs Rewards")
plt.subplot(2, 3, 4)
temp = PPO_arm_result['reward'].diff()/PPO_arm_result['reward']
print(temp)
count = 0
for i in range(len(temp)):
    if np.abs(temp[i]) < 0.001:
        if count == 10:
            break
        count += 1
plt.plot(PPO_arm_result['timestep'][:i],PPO_arm_result['reward'][:i])
plt.title("PPO Arm Env timestep vs Rewards")
plt.subplot(2, 3, 5)
temp = PPO_Pendulum_result['reward'].diff()/PPO_Pendulum_result['reward']
print(temp)
count = 0
for i in range(len(temp)):
    if np.abs(temp[i]) < 0.001:
        if count == 10:
            break
        count += 1
plt.plot(PPO_Pendulum_result['timestep'][:i],PPO_Pendulum_result['reward'][:i])
plt.title("PPO Pendulum Env timestep vs Rewards")
plt.subplot(2, 3, 6)
temp = PPO_cartpole_result['reward'].diff()/PPO_cartpole_result['reward']
print(temp)
count = 0
for i in range(len(temp)):
    if np.abs(temp[i]) < 0.001:
        if count == 10:
            break
        count += 1
plt.plot(PPO_cartpole_result['timestep'][:i],PPO_cartpole_result['reward'][:i])
plt.title("PPO Cartpole Env timestep vs Rewards")


plt.show()