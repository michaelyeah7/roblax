import gym
from gym import error, spaces, utils
from gym.utils import seeding
import math
import numpy as np
import jax.numpy as jnp
import os
import jax

from jbdl.rbdl.utils import ModelWrapper
from jax import device_put
from jbdl.rbdl.utils import xyz2int
from jbdl.rbdl.dynamics import forward_dynamics_core
from jbdl.rbdl.contact import detect_contact_core
from jbdl.rbdl.dynamics.state_fun_ode import dynamics_fun_extend_core, events_fun_extend_core
from jbdl.rbdl.dynamics import composite_rigid_body_algorithm_core
from jbdl.rbdl.contact.impulsive_dynamics import impulsive_dynamics_extend_core
from jbdl.rbdl.ode.solve_ivp import integrate_dynamics
from jax.custom_derivatives import closure_convert
import math
from jax.api import jit
from functools import partial
from jbdl.rbdl.tools import plot_model
from jbdl.experimental.ode.solve_ivp import solve_ivp

from jbdl.rbdl.tools import plot_model
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D

class HalfCheetahRBDLEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, render=True):
        action_max = np.ones(4)*10.
        self._action_space = spaces.Box(low=-action_max, high=action_max)
        observation_high = np.ones(14)*math.pi
        observation_low = np.zeros(14)
        self._observation_space = spaces.Box(low=observation_low, high=observation_high)

        self.seed()
        self.viewer = None

        if (render==True):
            plt.figure()
            plt.ion()

            fig = plt.gcf()
            self.ax = Axes3D(fig)


        #contruct the model

#         CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
        CURRENT_PATH = os.getcwd()
        print("CURRENT_PATH",CURRENT_PATH)
        # SCRIPTS_PATH = os.path.dirname(CURRENT_PATH)
        # print("SCRIPTS_PATH",SCRIPTS_PATH)
        MODEL_DATA_PATH = os.path.join(CURRENT_PATH, "jaxRBDL-2/scripts/model_data") 
        print("MODEL_DATA_PATH",MODEL_DATA_PATH)
        mdlw = ModelWrapper()
        mdlw.load(os.path.join(MODEL_DATA_PATH, 'half_max_v1.json'))
        model = mdlw.model
        self.model = model

        NC = int(model["NC"])
        NB = int(model["NB"])
        nf = int(model["nf"])
        contact_cond = model["contact_cond"]
        Xtree = device_put(model["Xtree"])
        ST = model["ST"]
        contactpoint = model["contactpoint"],
        idcontact = tuple(model["idcontact"])
        parent = tuple(model["parent"])
        jtype = tuple(model["jtype"])
        jaxis = xyz2int(model["jaxis"])
        contactpoint = model["contactpoint"]
        I = device_put(model["I"])
        a_grav = device_put(model["a_grav"])
        mu = device_put(0.9)
        contact_force_lb = device_put(contact_cond["contact_force_lb"])
        contact_force_ub = device_put(contact_cond["contact_force_ub"])
        contact_pos_lb = contact_cond["contact_pos_lb"]
        contact_vel_lb = contact_cond["contact_vel_lb"]
        contact_vel_ub = contact_cond["contact_vel_ub"]


        q0 = jnp.array([0.0,  0.4125, 0.0, math.pi/6, math.pi/6, -math.pi/3, -math.pi/3])
        qdot0 = jnp.zeros((7, ))

        q_star = jnp.array([0.0,  0.0, 0.0, math.pi/3, math.pi/3, -2*math.pi/3, -2*math.pi/3])
        qdot_star = jnp.zeros((7, ))

        x0 = jnp.hstack([q0, qdot0])
        t_span = (0.0, 2e-3)
        delta_t = 5e-4
        tau = 0.0

        ncp = 0

        def dynamics_fun(x, t, Xtree, I, contactpoint, u, a_grav, \
            contact_force_lb, contact_force_ub,  contact_pos_lb, contact_vel_lb, contact_vel_ub, mu,\
            ST, idcontact,   parent, jtype, jaxis, NB, NC, nf, ncp):
            q = x[0:NB]
            qdot = x[NB:]
            tau = jnp.matmul(ST, u)
            flag_contact = detect_contact_core(Xtree, q, qdot, contactpoint, contact_pos_lb, contact_vel_lb, contact_vel_ub,\
                idcontact, parent, jtype, jaxis, NC)
            xdot,fqp, H = dynamics_fun_extend_core(Xtree, I, q, qdot, contactpoint, tau, a_grav, contact_force_lb, contact_force_ub,\
            idcontact, flag_contact, parent, jtype, jaxis, NB, NC, nf, ncp, mu)
            return xdot

        def events_fun(y, t, Xtree, I, contactpoint, u, a_grav, contact_force_lb, contact_force_ub, \
            contact_pos_lb, contact_vel_lb, contact_vel_ub, mu, ST, idcontact,  parent, jtype, jaxis, NB, NC, nf, ncp):
            q = y[0:NB]
            qdot = y[NB:]
            flag_contact = detect_contact_core(Xtree, q, qdot, contactpoint, contact_pos_lb, contact_vel_lb, contact_vel_ub,\
                idcontact, parent, jtype, jaxis, NC)

            value = events_fun_extend_core(Xtree, q, contactpoint, idcontact, flag_contact, parent, jtype, jaxis, NC)
            return value

        def impulsive_dynamics_fun(y, t, Xtree, I, contactpoint, u, a_grav, contact_force_lb, contact_force_ub, \
            contact_pos_lb, contact_vel_lb, contact_vel_ub, mu, ST, idcontact,  parent, jtype, jaxis, NB, NC, nf, ncp):
            q = y[0:NB]
            qdot = y[NB:]
            H =  composite_rigid_body_algorithm_core(Xtree, I, parent, jtype, jaxis, NB, q)
            flag_contact = detect_contact_core(Xtree, q, qdot, contactpoint, contact_pos_lb, contact_vel_lb, contact_vel_ub,\
                idcontact, parent, jtype, jaxis, NC)
            qdot_impulse = impulsive_dynamics_extend_core(Xtree, q, qdot, contactpoint, H, idcontact, flag_contact, parent, jtype, jaxis, NB, NC, nf)
            qdot_impulse = qdot_impulse.flatten()
            y_new = jnp.hstack([q, qdot_impulse])
            return y_new

        t = device_put(0.0)

        pure_dynamics_fun = partial(dynamics_fun, ST=ST, idcontact=idcontact, \
                parent=parent, jtype=jtype, jaxis=jaxis, NB=NB, NC=NC, nf=nf, ncp=ncp)

        pure_events_fun = partial(events_fun, ST=ST, idcontact=idcontact, \
                parent=parent, jtype=jtype, jaxis=jaxis, NB=NB, NC=NC, nf=nf, ncp=ncp)

        pure_impulsive_fun =  partial(impulsive_dynamics_fun, ST=ST, idcontact=idcontact, \
            parent=parent, jtype=jtype, jaxis=jaxis, NB=NB, NC=NC, nf=nf, ncp=ncp)

        


        def _dynamics_step(y0, *pure_args):
            # t_eval, sol =  integrate_dynamics(pure_dynamics_fun, y0, t_span, delta_t, pure_events_fun, pure_impulsive_fun, args=args)
            # yT = sol[-1, :]
            t_eval = jnp.linspace(0, 2e-3, 4)
            xk = solve_ivp(pure_dynamics_fun, y0, t_eval, pure_events_fun, pure_impulsive_fun, *pure_args)[-1, :]
            # return yT
            return xk

        u = jnp.zeros((4,))
        self.pure_args = (Xtree, I, contactpoint, u, a_grav, contact_force_lb, contact_force_ub,  contact_pos_lb, contact_vel_lb, contact_vel_ub, mu)
        # print("pure_events",pure_events_fun(*self.pure_args))

        self.dynamics_step = _dynamics_step

    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self._observation_space

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # def _dynamics_step(self, y0, *args):
    #     t_eval, sol =  integrate_dynamics(pure_dynamics_fun, y0, t_span, delta_t, pure_events_fun, pure_impulsive_fun, args=args)
    #     yT = sol[-1, :]
    #     return yT

    def step(self, action):
        q_star = jnp.array([0.0,  0.0, 0.0, math.pi/6, -math.pi/6, -math.pi/3, math.pi/3])
        qdot_star = jnp.zeros((7, ))
        u = jnp.array(action)  # convert numpy.ndarray to jnp
        u = jnp.clip(u,-1,1)
        u = jnp.zeros((4,))
        # print("u",u)

        Xtree, I, contactpoint, u0, a_grav, contact_force_lb, contact_force_ub, contact_pos_lb, contact_vel_lb, contact_vel_ub,mu = self.pure_args
        pure_args = (Xtree, I, contactpoint, u, a_grav, contact_force_lb, contact_force_ub,  contact_pos_lb, contact_vel_lb, contact_vel_ub, mu)
        next_xk = self.dynamics_step(self.xk, *pure_args)
        # next_xk = jax.ops.index_update(next_xk,3,jnp.clip(next_xk[3], 0., math.pi/3))
        # next_xk = jax.ops.index_update(next_xk,4,jnp.clip(next_xk[4], -math.pi/3, 0.))
        # next_xk = jax.ops.index_update(next_xk,5,jnp.clip(next_xk[5], -math.pi/2, -math.pi/6))
        # next_xk = jax.ops.index_update(next_xk,6,jnp.clip(next_xk[6], math.pi/6, math.pi/2))
        # loss = jnp.sum((q_star[3:7] - next_xk[3:7])**2) + jnp.sum((qdot_star[3:7] - next_xk[10:14])**2)

        # qdot = next_xk[7:]
        # clipped_qdot = jnp.clip(qdot,-0.5,0.5)
        # next_xk = jax.ops.index_update(next_xk,jax.ops.index[7:],clipped_qdot)

        # next_xk = jnp.clip(next_xk,-2,2)
        # print("next_xk",next_xk)
        #refer to openai cheetah gym reward setting
        reward = np.array((next_xk[0] - self.xk[0])/2e-3 - 0.1 * jnp.square(u).sum())
        # reward = - np.array(loss)

        next_xk = np.array(next_xk)
        next_xk[5] = -math.pi/3
        # next_xk[5] = np.clip(next_xk[5],-math.pi/2, -math.pi/6)
        next_xk = jnp.array(next_xk)

        self.xk = next_xk
        self.state = np.array(self.xk) # update jnp state
        next_state = self.state # convert back to numpy.ndarray
        done = None
        return next_state, reward, done, {}

    def reset(self):
        self.xk = jnp.array([0.0,  0.4125, 0, math.pi/6, -math.pi/6, -math.pi/3, math.pi/3,0,0,0,0,0,0,0]) #xk refers to jaxRBDL state
        self.state = np.array(self.xk)
        return self.state
    ...
    # def render(self, mode='human', close=False):`
    # ...


    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
        plt.ioff()

    def plt_render(self):
        ax = self.ax
        ax.clear()
        plot_model(self.model, self.state[0:7], ax)
        # fcqp = np.array([0, 0, 1, 0, 0, 1])
        # plot_contact_force(model, xk[0:7], contact_force["fc"], contact_force["fcqp"], contact_force["fcpd"], 'fcqp', ax)
        ax.view_init(elev=0,azim=-90)
        ax.set_xlabel('X')
        # ax.set_xlim(-0.3, -0.3+0.6)
        ax.set_xlim(-0.3, -0.3+0.6)
        ax.set_ylabel('Y')
        ax.set_ylim(-0.15, -0.15+0.6)
        ax.set_zlabel('Z')
        ax.set_zlim(-0.1, -0.1+0.6)
        ax.set_title('Frame')
        plt.pause(1e-8)
