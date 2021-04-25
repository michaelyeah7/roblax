# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math

import gym
import jax
import jax.numpy as jnp
# from jax.ops import index_add
import numpy as np

from jaxRBDL.Dynamics.ForwardDynamics import ForwardDynamics, ForwardDynamicsCore
# from pyRBDL.Dynamics.ForwardDynamics import ForwardDynamics
from Simulator.UrdfWrapper import UrdfWrapper
# from jaxRBDL.Utils.UrdfWrapper_guo import UrdfWrapper
from Simulator.ObdlRender import ObdlRender
from Simulator.ObdlSim import ObdlSim
import os
import pybullet as p
from numpy import sin, cos

class Arm_rbdl():
    """
    Description:
        A 7 link arm robot contains 6 joints. The first base_link to arm_link_0 fixed joint 
        will be interpreted as prismatic joint (rbdl index 1) by rbdl. The remaining 5 joints are revolute
        joints (rbdl index 0).

    State:
        array of two jnp array       
        7-element array: Angles of 7 joints (the first one is a virtual prismatic joint transform from world to base).
        7-element array: Angle velocity of 7 joints.

    Actions:
        7-element List: forces applied on 7 joints.

    Reward:
        Square difference between current state and target. 

    Starting State:
        All zeros.

    Episode Termination:
        At least one angle is larger than 45 degrees.
    """

    def __init__(self, reward_fn=None, seed=0, render_flag=False):
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
        self.state_size = 14
        self.action_size = 7


        self.model = UrdfWrapper("urdf/arm.urdf").model
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
            qddot = ForwardDynamics(*input)
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

    def reset(self):
        # q = jax.random.uniform(
        #     self.random.get_key(), shape=(7,), minval=-0.05, maxval=0.05
        # )
        # qdot = jax.random.uniform(
        #     self.random.get_key(), shape=(7,), minval=-0.05, maxval=0.05
        # )

        # q = jnp.array(list(np.random.uniform(-0.05,0.05,7)))
        # qdot = jnp.array(list(np.random.uniform(-0.05,0.05,7)))
        q = jnp.zeros(7)
        qdot = jnp.zeros(7)
        self.state = jnp.array([q,qdot]).flatten()
        return self.state

    def step(self, state, action):
        self.state = self.dynamics(state, action)
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

        return reward


    def render(self):
        q, _ = jnp.split(self.state,2)
        # print("q for render",q)
        self.osim.step_theta(q)
