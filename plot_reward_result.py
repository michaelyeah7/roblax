import numpy as np
import pandas as pd
import math as math
import matplotlib.pyplot as plt


arm_result = pd.read_csv('arm_result.txt')
cartpole_result = pd.read_csv('SAC_cartpole_result.csv')
Pendulum_result = pd.read_csv('SAC_Pendulum_result.csv')

temp = Pendulum_result['Average Reward'].diff()/Pendulum_result['Average Reward']
print(temp)
count = 0
for i in range(len(temp)):
    if np.abs(temp[i]) < 0.001:
        if count == 10:
            break
        count += 1


plt.figure()
plt.plot(Pendulum_result['Timestep'][:i],Pendulum_result['Average Reward'][:i])
plt.show()