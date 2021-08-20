import numpy as np
import pandas as pd
import math as math
import matplotlib.pyplot as plt


arm_result = pd.read_csv('SAC_arm_result.csv')
cartpole_result = pd.read_csv('SAC_cartpole_result.csv')
Pendulum_result = pd.read_csv('SAC_Pendulum_result.csv')

temp = arm_result['Average Reward'].diff()/arm_result['Average Reward']
print(temp)
count = 0
for i in range(len(temp)):
    if np.abs(temp[i]) < 0.001:
        if count == 10:
            break
        count += 1


plt.figure()
plt.plot(arm_result['Timestep'][:i],arm_result['Average Reward'][:i])
plt.show()