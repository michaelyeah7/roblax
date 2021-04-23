import jax.numpy as jnp
import matplotlib.pyplot as plt
import jax
from jax import lax
from envs import Cartpole_rbdl, Cartpole_Hybrid
from agents import Deep_Cartpole_rbdl
import copy
import pickle
from time import gmtime, strftime 
from jaxRBDL.Dynamics.ForwardDynamics import ForwardDynamics, ForwardDynamicsCore
import numpy as np
import os
from model_based_RL import MBRL


# Init env and agent
lr = 1e-3
env = Cartpole_rbdl(render_flag=True) 
hybrid_env = Cartpole_Hybrid(model_lr=5e-1)
agent = Deep_Cartpole_rbdl(
             env_state_size = 4,
             action_space = jnp.array([0]),
             learning_rate = lr,
             gamma = 0.99,
             max_episode_length = 500,
             seed = 0
            )

# load_params = True
# update_params = False
# render = True

load_params = False
update_params = True


# loaded_params = pickle.load( open( "experiments2021-04-09 18:56:36/cartpole_svg_model_params_episode_130_2021-04-09 20:24:08.txt", "rb" ) )
# hybrid_env.model_params = loaded_params

if load_params == True:
    loaded_params = pickle.load( open( "experiments2021-04-12 20:42:23/cartpole_svg_value_params_episode_900_2021-04-13 13:51:06.txt", "rb" ) )
    agent.params = loaded_params

#init learner
mbrl = MBRL(env, agent)

episode_rewards = []
episodes_num = 1000
T = 500
horizon = 50
exp_dir = "experiments_lr_%d_horizon_%d_" % (lr, horizon)+ strftime("%Y-%m-%d %H:%M:%S", gmtime())
os.mkdir(exp_dir)



for j in range(episodes_num):

    rewards = 0
    env.reset()           
    print("episode:{%d}" % j)

    #evaluate rewards and update value function
    rewards, trajectory_state_buffer = mbrl.roll_out_for_render(env, hybrid_env, agent, agent.params, T)
    # rewards, trajectory_state_buffer = mbrl.roll_out_for_render(env, hybrid_env, agent, (agent.params, agent.rnn_params), T)

    #update the policy
    if (update_params==True):
        #update policy using 20 horizon 5 partial trajectories
        # for i in range(20):
        env.reset()
        # hybrid_env.reset() 

        random_state_index = np.random.randint(len(trajectory_state_buffer), size=1)[0]
        # print("random_state_index",random_state_index)
        # print("trajectory_state_buffer[random_state_index]",trajectory_state_buffer.shape)
        env.state =  trajectory_state_buffer[random_state_index]

        #train policy use 5-step partial trajectory and learned value function
        total_return, grads = mbrl.f_grad(env, agent, (agent.params, agent.value_params), T)
        # total_return, grads = mbrl.f_grad(env, agent, (agent.params, agent.rnn_params), T)
        # total_return, grads = mbrl.f_grad(env, agent, agent.params, T)
        # total_return, grads = mbrl.f_grad(hybrid_env, agent, (agent.params, agent.value_params),T)

        #get and update policy and value function grads
        policy_grads, value_grads = grads 
        # policy_grads, rnn_grads = grads 
        # policy_grads = grads 
        # print("policy_grads",policy_grads)         
        agent.params = agent.update(policy_grads, agent.params, agent.lr)
        agent.value_params =  agent.update(value_grads,agent.value_params, agent.lr)
        # agent.rnn_params = agent.update(rnn_grads, agent.rnn_params, agent.lr)
        # agent.state = jnp.zeros((agent.env_state_size,))
        # agent.h_t = jnp.zeros(4)

    episode_rewards.append(rewards)
    print("rewards is %f" % rewards)
    # print("hybrid_env.model_losses is %f" % hybrid_env.model_losses[j])
    # if (j%10==0 and j!=0 and update_params==True):
    if (j%10==0 and j!=0):
        #for agent loss
        with open(exp_dir + "/cartpole_svg_params"+ "_episode_%d_" % j +".txt", "wb") as fp:   #Pickling
            pickle.dump(agent.params, fp)
        # with open(exp_dir + "/cartpole_rnn_params"+ "_episode_%d_" % j  +".txt", "wb") as fp:   #Pickling
        #     pickle.dump(agent.rnn_params, fp)
        plt.figure()
        plt.plot(episode_rewards[1:])
        plt.savefig((exp_dir + '/cartpole_svg_loss_episode_%d_' % j)+ strftime("%Y-%m-%d %H:%M:%S", gmtime()) + '.png')
        plt.close()

        #for value function loss
        with open(exp_dir + "/cartpole_svg_value_params"+ "_episode_%d_" % j + strftime("%Y-%m-%d %H:%M:%S", gmtime()) +".txt", "wb") as fp:   #Pickling
            pickle.dump(agent.value_params, fp)
        plt.figure()
        plt.plot(agent.value_losses)
        plt.savefig((exp_dir + '/cartpole_svg_agent_value_loss_episode_%d_' % j) + strftime("%Y-%m-%d %H:%M:%S", gmtime()) + '.png')
        plt.close()        
        # #for model loss
        # with open(exp_dir + "/cartpole_svg_model_params"+ "_episode_%d_" % j + strftime("%Y-%m-%d %H:%M:%S", gmtime()) +".txt", "wb") as fp:   #Pickling
        #     pickle.dump(hybrid_env.model_params, fp)
        # plt.figure()
        # plt.plot(hybrid_env.model_losses)
        # plt.savefig((exp_dir + '/cartpole_svg_model_loss_episode_%d_' % j) + strftime("%Y-%m-%d %H:%M:%S", gmtime()) + '.png')
        # plt.close()
