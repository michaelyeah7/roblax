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


def loop(context, x):
    env, agent, params = context
    control = agent(env.state, params)
    prev_state = copy.deepcopy(env.state)
    _, reward, done, _ = env.step(env.state,control)

    return (env, agent), reward, done

def roll_out(env, agent, params, T):
    policy_params, value_params =  params
    gamma = 0.9
    total_return = 0.0
    for i in range(5):
        (env, agent), r, done= loop((env, agent,policy_params), i)
        total_return = total_return * gamma + r 
        if done:
            print("end this episode because out of threshhold in policy update")
            env.past_reward = 0
            break
    total_return += agent.value(env.state,value_params) * gamma 
    losses = -total_return       
    return losses

f_grad = jax.value_and_grad(roll_out,argnums=2)
# value_grad = jax.value_and_grad(roll_out,argnums=3)

def loss_value(state, next_state, reward, value_params):
    td = reward + agent.value(next_state, value_params) - agent.value(state, value_params)
    value_loss = 0.5 * (td ** 2)
    return value_loss

value_loss_grad = jax.value_and_grad(loss_value,argnums=3)

def loss_hybrid_model(prev_state, control, true_next_state, model_params):
    next_state = hybrid_env.forward(prev_state, control, model_params)
    model_loss = jnp.sum((next_state - true_next_state)**2)
    # model_loss = jnp.linalg.norm(next_state - true_next_state)
    # print("model loss",model_loss)
    # print("model_loss.value",model_loss[0])
    # model_losses.append(model_loss)
    return model_loss
# model_loss_grad = jax.grad(loss_hybrid_model,argnums=3)
model_loss_grad = jax.value_and_grad(loss_hybrid_model,argnums=3)

def loop_for_render(context, x):
    env, hybrid_env, agent, params = context
    if(render==True):
        env.osim_render()
    control = agent(env.state, params)
    prev_state = copy.deepcopy(env.state)
    next_state, reward, done, _ = env.step(env.state,control)

    #update value function
    value_loss, value_grads =  value_loss_grad(prev_state,next_state,reward,agent.value_params)
    agent.value_losses.append(value_loss)
    agent.value_params = agent.update(value_grads,agent.value_params,agent.lr)    
    
    # #update hybrid model
    # model_loss, model_grads = model_loss_grad(prev_state,control,next_state,hybrid_env.model_params)
    # # print("model_loss",model_loss)
    # hybrid_env.model_losses.append(model_loss)
    # hybrid_env.model_params = agent.update(model_grads,hybrid_env.model_params,hybrid_env.model_lr)


    return (env, hybrid_env, agent), reward, done

def roll_out_for_render(env, hybrid_env, agent, params, T):
    gamma = 0.9
    rewards = 0.0
    for i in range(T):
        (env, hybrid_env, agent), r, done= loop_for_render((env, hybrid_env, agent,params), i)
        rewards = rewards * gamma + r 
        if done:
            print("end this episode because out of threshhold in model update")
            env.past_reward = 0
            break
        
    return rewards



# Deep
env = Cartpole_rbdl() 
hybrid_env = Cartpole_Hybrid(model_lr=1e-1)
agent = Deep_Cartpole_rbdl(
             env_state_size = 4,
             action_space = jnp.array([0]),
             learning_rate = 0.5,
             gamma = 0.99,
             max_episode_length = 500,
             seed = 0
            )

# load_params = True
# update_params = False
# render = True

load_params = False
update_params = True
render = True

if load_params == True:
    loaded_params = pickle.load( open( "examples/cartpole_svg_params_episode_100_2021-04-05 06:10:53.txt", "rb" ) )
    agent.params = loaded_params

 # for loop version
# xs = jnp.array(jnp.arange(T))
print(env.reset())
rewards = 0
episode_rewards = []
# episodes_num = 1000
episodes_num = 1000
T = 100
# T = 1000
for j in range(episodes_num):

    rewards = 0
    env.reset()           
    print("episode:{%d}" % j)

    #update hybrid model using real trajectories
    rewards = roll_out_for_render(env, hybrid_env, agent, agent.params, T)
    # print("hybrid_env.model_losses",hybrid_env.model_losses)

    #update the parameter
    if (update_params==True):
        #update policy using 20 horizon 5 partial trajectories
        for i in range(20):
            env.reset()
            # hybrid_env.reset() 

            #train agent using learned hybrid env
            total_return, grads = f_grad(env, agent, (agent.params, agent.value_params), T)
            policy_grads, value_grads = grads
            # _, value_params_grads = value_grad(env, agent, agent.params, agent.value_params, T)
            # total_return, grads = f_grad(hybrid_env, agent, agent.params,T)
            
            agent.params = agent.update(policy_grads, agent.params, agent.lr)
            agent.value_params =  agent.update(value_grads,agent.value_params, agent.lr)

    episode_rewards.append(rewards)
    print("rewards is %f" % rewards)
    if (j%10==0 and j!=0 and update_params==True):
        #for agent loss
        with open("examples/cartpole_svg_params"+ "_episode_%d_" % j + strftime("%Y-%m-%d %H:%M:%S", gmtime()) +".txt", "wb") as fp:   #Pickling
            pickle.dump(agent.params, fp)
        plt.figure()
        plt.plot(episode_rewards[1:])
        plt.savefig(('cartpole_svg_loss_episode_%d_' % j)+ strftime("%Y-%m-%d %H:%M:%S", gmtime()) + '.png')
        plt.close()
        #for value function loss
        with open("examples/cartpole_svg_value_params"+ "_episode_%d_" % j + strftime("%Y-%m-%d %H:%M:%S", gmtime()) +".txt", "wb") as fp:   #Pickling
            pickle.dump(agent.value_params, fp)
        plt.figure()
        plt.plot(agent.value_losses)
        plt.savefig(('cartpole_svg_agent_value_loss_episode_%d_' % j) + strftime("%Y-%m-%d %H:%M:%S", gmtime()) + '.png')
        plt.close()        
        # #for model loss
        # with open("examples/cartpole_svg_model_params"+ "_episode_%d_" % j + strftime("%Y-%m-%d %H:%M:%S", gmtime()) +".txt", "wb") as fp:   #Pickling
        #     pickle.dump(hybrid_env.model_params, fp)
        # plt.figure()
        # plt.plot(hybrid_env.model_losses)
        # plt.savefig(('cartpole_svg_model_loss_episode_%d_' % j) + strftime("%Y-%m-%d %H:%M:%S", gmtime()) + '.png')
        # plt.close()

# fp.close()