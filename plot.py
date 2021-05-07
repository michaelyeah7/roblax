import pickle
import matplotlib.pyplot as plt
import math
import numpy as np
import seaborn as sns

def plot_pendulum():
    file = "examples/models/pendulum/" + "pendulum_rewards_episode_200_" + ".txt"
    with open(file, "rb") as f:
        ours_data = pickle.load(f)
    x1 = ours_data
    x1 = ours_data[:40]
    x1 = np.asarray(x1)
    # x1 = np.asarray(x1).reshape(-1,10).T

    #PILCO
    file = "plotting_data/" + "PILCO_episode_reward_list_episode_25_" + ".txt"
    with open(file, "rb") as f:
        PILCO_data = pickle.load(f)
    x2 = PILCO_data
    x2 = np.asarray(x2)
    x2 = np.concatenate((x2,np.ones(14)*(-388.53)))
    # x2 = x2.reshape(-1,10).T

    #MBPO
    file = "plotting_data/" + "MBPO_pendulum_rewards_episode_43_" + ".txt"
    with open(file, "rb") as f:
        MBPO_data = pickle.load(f)
    x3 = MBPO_data[:40]
    x3 = np.asarray(x3)
    # x3 = np.concatenate((x3,np.ones(57)*(-300.0)))
    # x3 = x2.reshape(-1,10).T

    #DDPG
    file = "plotting_data/" + "DDPG_pendulum_rewards_episode_800_" + ".txt"
    with open(file, "rb") as f:
        DDPG_data = pickle.load(f)
    x4 = DDPG_data
    x4 = np.asarray(x4)
    # x4 = np.concatenate((x4,np.ones(57)*(-300.0)))
    # x4 = x4.reshape(-1,10).T

    time = np.asarray(range(x1.shape[0])) 

    sns.set(style="darkgrid", font_scale=1.0)
    sns.tsplot(time=time, data=x1, color="r", condition="Ours")
    sns.tsplot(time=time, data=x2, color="b", condition="PILCO")
    sns.tsplot(time=time, data=x3, color="g", condition="MBPO")
    sns.tsplot(time=time, data=x4, color="yellow", condition="DDPG")
    # sns.tsplot(time=time, data=x2, color="b", condition="dagger")

    plt.ylabel("Reward")
    plt.xlabel("Episode")
    plt.title("Pendulum")
    plt.savefig('examples/models/pendulum/pendulum_episode_%d_' % 100 + '.png')
    plt.close()

def plot_cartpole():
    file = "examples/models/cartpole/" + "cartpole_rewards_episode_1210_" + ".txt"
    with open(file, "rb") as f:
        data = pickle.load(f)

    x1 = data[:1200]
    x1 = np.asarray(x1).reshape(-1,50).T
    time = np.asarray(range(x1.shape[1])) * 50

    sns.set(style="darkgrid", font_scale=1.0)
    sns.tsplot(time=time, data=x1, color="r", condition="cartpole reward")
    # sns.tsplot(time=time, data=x2, color="b", condition="dagger")

    plt.ylabel("Reward")
    plt.xlabel("Episode")
    plt.title("Cartpole Rewards")
    plt.savefig('examples/models/cartpole/cartpole_svg_agent_value_loss_episode_%d_' % 1210 + '.png')
    plt.close()

def plot_arm():
    file = "examples/models/arm/" + "arm_rewards_episode_230_" + ".txt"
    with open(file, "rb") as f:
        data = pickle.load(f)

    x1 = data
    x1 = np.asarray(data).reshape(-1,10).T
    time = np.asarray(range(x1.shape[1])) * 10

    sns.set(style="darkgrid", font_scale=1.0)
    sns.tsplot(time=time, data=x1, color="r", condition="arm reward")
    # sns.tsplot(time=time, data=x2, color="b", condition="dagger")

    plt.ylabel("Reward")
    plt.xlabel("Episode")
    plt.title("Arm Rewards")
    plt.savefig('examples/models/arm/arm_svg_agent_value_loss_episode_%d_' % 230 + '.png')
    plt.close()


if __name__ == '__main__':
    plot_pendulum()
    # plot_cartpole()
    # plot_arm()