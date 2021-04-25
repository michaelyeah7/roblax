import pickle
import matplotlib.pyplot as plt
import math
import numpy as np
import seaborn as sns


file = "pendulum_experiments_lr_0.000010_horizon_100_2021-04-25 01:57:03/" + "pendulum_rewards_episode_200_" + ".txt"
with open(file, "rb") as f:
    data = pickle.load(f)

x1 = data
x1 = np.asarray(data).reshape(10,-1)
time = range(x1.shape[1])

sns.set(style="darkgrid", font_scale=1.0)
sns.tsplot(time=time, data=x1, color="r", condition="pendulum reward")
# sns.tsplot(time=time, data=x2, color="b", condition="dagger")

plt.ylabel("Reward")
plt.xlabel("Episode")
plt.title("Pendulum")
plt.savefig('examples/models/pendulum/pendulum_svg_agent_value_loss_episode_%d_' % 200 + '.png')
plt.close()