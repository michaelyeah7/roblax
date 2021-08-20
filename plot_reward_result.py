import numpy as np
import pandas as pd
import math as math
import matplotlib.pyplot as plt


arm_result = pd.read_csv('SAC_arm_result.csv')
cartpole_result = pd.read_csv('SAC_cartpole_result.csv')
Pendulum_result = pd.read_csv('tmp_data/SAC_jbdl_pendulum-v0_log_11.csv')

temp = Pendulum_result['reward'].diff()/Pendulum_result['reward']
print(temp)
count = 0
for i in range(len(temp)):
    if np.abs(temp[i]) < 0.001:
        if count == 10:
            break
        count += 1


plt.figure()
plt.plot(Pendulum_result['timestep'][:i],Pendulum_result['reward'][:i])
plt.show()