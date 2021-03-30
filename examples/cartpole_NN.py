import jax.numpy as jnp
import matplotlib.pyplot as plt
import jax
from jax import lax
from envs import CartPole, Cartpole_rbdl
from agents import Deep_Cartpole
import copy
import pickle
from time import gmtime, strftime 
from jaxRBDL.Dynamics.ForwardDynamics import ForwardDynamics, ForwardDynamicsCore

def forward(state, w, env, agent):
    action = agent.__call__(state, w)



    # x, x_dot, theta, theta_dot = state
    # force = jnp.clip(action * 100,-10,10)
    # # force = action
    # q = jnp.array([0,x,theta])
    # qdot = jnp.array([0,x_dot,theta_dot])
    # torque = jnp.array([0,force,0.0])
    # # input = (env.model, q, qdot, torque)
    # model = env.model

    # input = (model["Xtree"], model["I"], tuple(model["parent"]), tuple(model["jtype"]), model["jaxis"],
    #         model["NB"], q, qdot, torque, model["a_grav"])
    # accelerations = ForwardDynamicsCore(*input)

    # xacc = accelerations[1][0]
    # thetaacc = accelerations[2][0]

    # tau = 0.02
    # x = x + tau * x_dot
    # x_dot = x_dot + tau * xacc
    # theta = theta + tau * theta_dot
    # theta_dot = theta_dot + tau * thetaacc

    # next_state = jnp.array([x, x_dot, theta, theta_dot])




    next_state = env.dynamics(state,action)
    # next_state, reward, done, _ = env.step(state,action)
    reward = env.reward_func(next_state)
    return reward

# f_grad = jax.grad(forward,argnums=1)
f_grad = jax.jacfwd(forward,argnums=1)


def loop(context, x):
    env, agent = context
    if(render==True):
        env.render()
    control = agent(env.state, agent.params)
    prev_state = copy.deepcopy(env.state)
    # print("control",control)
    # _, reward, done, _ = env.step(control)
    # reward = forward(env.state, agent.params, env, agent)
    _, reward, done, _ = env.step(env.state,control)
    # agent.feed(reward)
    # agent.update()
    # print("prev_state",prev_state)
    if (update_params==True):
        grads = f_grad(prev_state, agent.params, env, agent)
        # print("grads",grads)
        agent.params = [(w - agent.lr * dw, b - agent.lr * db)
                for (w, b), (dw, db) in zip(agent.params, grads)]
    # print("agent.params",agent.params)

    # agent.W -= agent.lr * d_reward_d_w
    # print("agent.W",agent.W)
    return (env, agent), reward, done

# Deep
env = CartPole()
agent = Deep_Cartpole(
             env_state_size = 4,
             action_space = jnp.array([0,1]),
             learning_rate = 0.1,
             gamma = 0.99,
             max_episode_length = 500,
             seed = 0
            )
# loaded_params = pickle.load( open( "examples/cartpole_params2021-02-21 06:45:33.txt", "rb" ) )

# agent.params = loaded_params

 # for loop version
# xs = jnp.array(jnp.arange(T))
print(env.reset())
update_params = True
render = True
reward = 0
loss = 0
episode_loss = []
episodes_num = 100
T = 200
for j in range(episodes_num):

    loss = 0
    env.reset()           
    print("episode:{%d}" % j)
    for i in range(T):
        (env, agent), r, done= loop((env, agent), 0)
        loss += r
        if done:
            print("end this episode because out of threshhold")
            env.past_reward = 0
            break
        # print("loss:",r)
        # loss.append(r)
        # reward += r
    episode_loss.append(loss)
    print("loss is %f and lasts for %d steps" % (loss,i))
# reward_forloop = reward
# print('reward_forloop = ' + str(reward_forloop))
plt.plot(episode_loss[1:])

#save plot and params
plt.savefig('cartpole_rbdl_loss'+ strftime("%Y-%m-%d %H:%M:%S", gmtime()) + '.png')
with open("params"+ strftime("%Y-%m-%d %H:%M:%S", gmtime()) +".txt", "wb") as fp:   #Pickling
    pickle.dump(agent.params, fp)