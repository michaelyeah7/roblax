import gym
from gym import error, spaces, utils
from gym.utils import seeding

import math

import gym
import jax
import jax.numpy as jnp
import numpy as np

# from utils import Random
from jaxRBDL.Dynamics.ForwardDynamics import ForwardDynamics, ForwardDynamicsCore
# from pyRBDL.Dynamics.ForwardDynamics import ForwardDynamics
from simulator.UrdfWrapper import UrdfWrapper
# from jaxRBDL.Utils.UrdfWrapper_guo import UrdfWrapper
from simulator.ObdlRender import ObdlRender
from simulator.ObdlSim import ObdlSim
import os
import pybullet as p
from numpy import sin, cos

class CartpoleRBDLEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, render=True):
        action_max = np.ones(1)*100.
        self._action_space = spaces.Box(low=-action_max, high=action_max)
        observation_high = np.ones(4)*math.pi
        observation_low = np.zeros(4)
        self._observation_space = spaces.Box(low=observation_low, high=observation_high)

        self.seed()
        self.viewer = None

        self.tau = 0.02
        # self.model = UrdfWrapper("urdf/inverted pendulum_link1_1.urdf").model
        self.model = UrdfWrapper("urdf/cartpole_add_base.urdf").model
        self.osim = ObdlSim(self.model,dt=self.tau,vis=True)

        
        self.l = 4.0

    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self._observation_space

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        x, x_dot, theta, theta_dot = self.state
        force = action[0]

        q = jnp.array([0,0,x,theta])
        qdot = jnp.array([0,0,x_dot,theta_dot])
        torque = jnp.array([0,0,force,0.])
        # print("q",q)
        # print("qdot",qdot)
        # print("force",force)
        input = (self.model, q, qdot, torque)
        accelerations = ForwardDynamics(*input)
        # print("accelerations",accelerations)
        # xacc = accelerations[2][0]
        # thetaacc = accelerations[3][0]
        xacc = accelerations[2]
        thetaacc = accelerations[3]    


        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc

        #state
        next_state = jnp.array([x, x_dot, theta, theta_dot])
        next_state =  np.array(next_state)
        self.state = next_state

        #reward
        A = 1
        invT = A * jnp.array([[1, self.l, 0], [self.l, self.l ** 2, 0], [0, 0, self.l ** 2]])
        j = jnp.array([x, jnp.sin(theta), jnp.cos(theta)])
        j_target = np.array([0.0, 0.0, 1.0])

        reward = jnp.matmul((j - j_target), invT)
        reward = jnp.matmul(reward, (j - j_target))
        reward = -(1 - jnp.exp(-0.5 * reward))
        reward = np.array(reward)
        # print("reward",reward)

        done = False

        return next_state, reward, done, {}

    def reset(self):
        self.state = np.array([0., 0., 3.14, 0.])
        return self.state

    def osim_render(self):
        q = [0,0,self.state[0],self.state[2]]
        self.osim.step_theta(q)
        

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
        plt.ioff()