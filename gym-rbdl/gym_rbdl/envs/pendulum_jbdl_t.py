import math

import gym
import jax
import jax.numpy as jnp
# from jax.ops import index_add
import numpy as np

from gym import error, spaces, utils
from gym.utils import seeding

# from jaxRBDL.Dynamics.ForwardDynamics import ForwardDynamics, ForwardDynamicsCore
from jbdl.rbdl.dynamics.forward_dynamics import  forward_dynamics, forward_dynamics_core
# from pyRBDL.Dynamics.ForwardDynamics import ForwardDynamics
from simulator.UrdfWrapper import UrdfWrapper
# from jaxRBDL.Utils.UrdfWrapper_guo import UrdfWrapper
from simulator.ObdlRender import ObdlRender
from simulator.ObdlSim import ObdlSim
import os
import pybullet as p
from numpy import sin, cos
import time

class PendulumJBDLEnv(gym.Env):
    def __init__(self, reward_fn=None, seed=0, render_flag=False):


        action_max = np.ones(1)*100.
        self._action_space = spaces.Box(low=-action_max, high=action_max)
        observation_high = np.ones(1)*math.pi
        observation_low = np.zeros(1)
        self._observation_space = spaces.Box(low=observation_low, high=observation_high)



        self.tau = 0.01  # seconds between state updates
        self.kinematics_integrator = "euler"
        self.viewer = None
        self.target = jnp.array([0,0,0,0,1.57,0,0])
        self.qdot_target = jnp.zeros(7)
        self.qdot_threshold = 100.0
        # Angle at which to fail the episode
        # Angle at which to fail the episode
        self.theta_threshold_radians = math.pi / 2
        # self.x_threshold = 2.4


        self.model = UrdfWrapper("urdf/inverted pendulum_link1_1.urdf").model
        # self.model = UrdfWrapper("urdf/two_link_arm.urdf").model
        self.osim = ObdlSim(self.model,dt=self.tau,vis=True)
        self.render_flag = render_flag
        
        self.reset()

        # @jax.jit
        def _dynamics(state, action):
            q, qdot = jnp.split(state, 2)
            torque = action/10
            # torque = action * 100
            # torque = jnp.array(action)
            # print("q",q)
            # print("qdot",qdot)
            # print("torque",torque)
            input = (self.model, q, qdot, torque)
            #ForwardDynamics return shape(NB, 1) array
            qddot = forward_dynamics(*input)
            qddot = qddot.flatten()
            # qddot = jnp.clip(qddot,0,0.5)
            # print("qddot",qddot)

             
            for i in range(2,len(q)-1):
                qdot = jax.ops.index_add(qdot, i, self.tau * qddot[i])
                q = jax.ops.index_add(q, i, self.tau * qdot[i]) 
            # qdot = jnp.zeros(7) 
            # print("q[5]",q[5])
            # print("qddot",qddot)
            # print("qdot",qdot)

            return jnp.array([q, qdot]).flatten()
        
        self.dynamics = _dynamics

    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self._observation_space

    def reset(self):
        # q = jax.random.uniform(
        #     self.random.get_key(), shape=(7,), minval=-0.05, maxval=0.05
        # )
        # qdot = jax.random.uniform(
        #     self.random.get_key(), shape=(7,), minval=-0.05, maxval=0.05
        # )

        q = jnp.array(list(np.random.uniform(-0.05,0.05,9)))
        qdot = jnp.array(list(np.random.uniform(-0.05,0.05,9)))
        self.state = jnp.array([q,qdot]).flatten()
        self.state =  np.array(self.state)
        return self.state

    def step(self, action):
        self.state = self.dynamics(self.state, action)
        q, qdot = jnp.split(self.state, 2)

        # done = jax.lax.cond(
        #     (jnp.abs(x) > jnp.abs(self.x_threshold))
        #     + (jnp.abs(theta) > jnp.abs(self.theta_threshold_radians)),
        #     lambda done: True,
        #     lambda done: False,
        #     None,
        # )

        reward = self.reward_func(self.state)

        done = False
        if (len(qdot[qdot>self.qdot_threshold]) >0):
            # print("q in done",q)
            done = True
            reward += 10


        self.state =  np.array(self.state)
        return self.state, reward, done, {}


    def reward_func(self,state):
        # # x, x_dot, theta, theta_dot = state
        # reward = state[0]**2 + (state[1])**2 + 100*state[2]**2 + state[3]**2 
        # # reward = jnp.exp(state[0])-1 + state[2]**2 + state[3]**2 
        q, qdot = jnp.split(state,2)
        # print("q in reward",q)
        # print("qdot in reward", qdot)
        # reward = jnp.log(jnp.sum(jnp.square(q - self.target))) + jnp.log(jnp.sum(jnp.square(qdot - self.qdot_target))) 
        costs = jnp.log(jnp.sum(jnp.square(q - self.target))) 
        # reward = jnp.log((q[5]-1.57)**2) + jnp.log(jnp.sum(jnp.square(qdot - self.qdot_target)))
        # reward = jnp.linalg.norm(jnp.square(q - self.target)) + jnp.linalg.norm(jnp.square(qdot - self.qdot_target))
        reward = -costs
        reward = np.array(reward)
        return reward


    def osim_render(self):
        q, _ = jnp.split(self.state,2)
        # print("q for render",q)
        self.osim.step_theta(q)

if __name__ == '__main__':
    inverted_pendulum = Inverted_Pendulum()
    time.sleep(10)
    # hopper.render()