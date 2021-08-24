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
from simulator.UrdfWrapper import UrdfWrapper
# from jaxRBDL.Utils.UrdfWrapper_guo import UrdfWrapper
from simulator.ObdlRender import ObdlRender
from simulator.ObdlSim import ObdlSim
import os
import pybullet as p
from numpy import sin, cos
import time

from jaxRBDL.Dynamics.CompositeRigidBodyAlgorithm import CompositeRigidBodyAlgorithm
from jaxRBDL.Dynamics.InverseDynamics import InverseDynamics
from jax.numpy.linalg import inv
from jaxRBDL.Contact.DetectContact import DetectContact
from jaxRBDL.Contact.CalcContactForceDirect import CalcContactForceDirect
from jaxRBDL.Contact.SolveContactLCP import SolveContactLCP
# from jaxRBDL.Dynamics.ForwardDynamics import ForwardDynamics
from jaxRBDL.Kinematics.CalcBodyToBaseCoordinates import CalcBodyToBaseCoordinates


from jbdl.rbdl.dynamics.state_fun_ode import dynamics_fun_core, dynamics_fun
from jbdl.rbdl.contact.detect_contact import detect_contact
from jax import device_put

class Hopper():

    def __init__(self, reward_fn=None, seed=0, render_flag=False):
        self.t = 0.01  # seconds between state updates
        self.time = 0.
        self.kinematics_integrator = "euler"
        self.viewer = None
        self.target = jnp.array([0,0,0,0,0])
        self.qdot_target = jnp.zeros(5)
        self.qdot_threshold = 100.0
        # Angle at which to fail the episode
        # Angle at which to fail the episode
        self.theta_threshold_radians = math.pi / 2
        # self.x_threshold = 2.4


        model = UrdfWrapper("urdf/hopper.urdf").model
        # self.model = UrdfWrapper("urdf/two_link_arm.urdf").model

        # model = UrdfWrapper("urdf/laikago/laikagolow.urdf").model
        model["jtype"] = jnp.asarray(model["jtype"])
        model["parent"] = jnp.asarray(model["parent"])
        # model["H"] = CompositeRigidBodyAlgorithm(model, q)
        # model["C"] = InverseDynamics(model, q, qdot, jnp.zeros((NB, 1)))
        # model["Hinv"] = inv(model["H"])
        model["nf"] = 2

        contact_cond = dict()
        contact_cond["contact_pos_lb"] = jnp.array([0.0001, 0.0001, 0.0001]).reshape(-1, 1)
        contact_cond["contact_pos_ub"] = jnp.array([0.0001, 0.0001, 0.0001]).reshape(-1, 1)
        contact_cond["contact_vel_lb"] = jnp.array([-0.05, -0.05, -0.05]).reshape(-1, 1)
        contact_cond["contact_vel_ub"] = jnp.array([0.01, 0.01, 0.01]).reshape(-1, 1)
        contact_cond["contact_force_lb"] = np.array([[-1000.0], [-1000.0], [0.]]).reshape(-1, 1)
        contact_cond["contact_force_ub"] = np.array([[1000.0], [1000.0], [3000.0]]).reshape(-1, 1)

        model["contact_cond"] = contact_cond
        # _model["contact_cond"] = contact_cond
        model["contactpoint"] = [np.array([[0.],[0.],[0]])]
        model["idcontact"] = [3]
        model["NC"] = 1
        # _model["idcontact"] = jnp.array([2]).reshape(-1, 1)

        # model['tau'] = torque 
        model['ST'] = jnp.zeros((3,)) # useless

        self.model = model

        self.osim = ObdlSim(self.model,dt=self.t,vis=True)
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


            X = jnp.hstack((q,qdot))
            x = X
            model = self.model


            NB = int(model["NB"])
            NC = int(model["NC"])
            # Get q qdot tau
            q = X[0:NB]
            qdot = X[NB: 2 * NB]

            model["tau"] = torque
            # Calcualte H C 
       
            #forward dynamics
            # T = self.t
            
            # Calculate contact force in joint space
            # flag_contact = DetectContact(model, q, qdot, contact_cond)
            # flag_contact_tuple = detect_contact(model, q, qdot)
            # flag_contact_list = []
            # flag_contact_list.append(flag_contact_tuple)
            # flag_contact = jnp.array(flag_contact_list).flatten()
            # print("flag_contact",flag_contact)

            contact_force = dict()
            xdot = dynamics_fun(self.time, x, model, contact_force)
            print("xdot",xdot)
            x = xdot + x
            q, qdot = jnp.split(x,2)

            self.time += self.t

            # # print("In Dynamics!!!")
            # if jnp.sum(flag_contact) !=0: 
            #     # lambda, fqp, fpd] = SolveContactLCP(q, qdot, tau, flag_contact);
            #     # lam, fqp, fc, fcqp, fcpd = CalcContactForceDirect(_model, q, qdot, tau, flag_contact, 2)
            #     lam, fqp, fc, fcqp, fcpd = SolveContactLCP(_model, q, qdot, tau, flag_contact,0.1)
            #     contact_force["fc"] = fc
            #     contact_force["fcqp"] = fcqp
            #     contact_force["fcpd"] = fcpd
            # else:
            #     # print("No Conatact")
            #     lam = jnp.zeros((NB, 1))
            #     contact_force["fc"] = jnp.zeros((3*NC, 1))
            #     contact_force["fcqp"] = jnp.zeros((3*NC, 1))
            #     contact_force["fcpd"] = jnp.zeros((3*NC, 1))


            # # Forward dynamics
            # Tau = tau + lam
            # qddot = ForwardDynamics(model, q, qdot, Tau).flatten()
            # input = (self.model, q, qdot, torque)
            # #ForwardDynamics return shape(NB, 1) array
            # qddot = ForwardDynamics(*input)
            # qddot = qddot.flatten()

            # qdot_hat = qdot + qddot * self.t
            # q_hat = q + qdot * self.t


            # return jnp.array([q_hat, qdot_hat]).flatten()
            return jnp.array([q, qdot]).flatten()
        
        self.dynamics = _dynamics

    def reset(self):
        # q = jax.random.uniform(
        #     self.random.get_key(), shape=(7,), minval=-0.05, maxval=0.05
        # )
        # qdot = jax.random.uniform(
        #     self.random.get_key(), shape=(7,), minval=-0.05, maxval=0.05
        # )

        q = jnp.array(list(np.random.uniform(-0.05,0.05,5)))
        qdot = jnp.array(list(np.random.uniform(-0.05,0.05,5)))
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

if __name__ == '__main__':
    hopper = Hopper()
    action = jnp.array([0.0])
    for i in range(10):
        print("timestep i",i)
        hopper.step(hopper.state, action)
        # print("hopper.state",hopper.state)
        time.sleep(0.1)
        hopper.render()