import gym
from gym import error, spaces, utils
from gym.utils import seeding
import math
import numpy as np
import jax.numpy as jnp
import os

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

class HalfCheetahRBDLEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        action_max = np.ones(4)*10.
        self._action_space = spaces.Box(low=-action_max, high=action_max)
        observation_high = np.ones(14)*math.pi
        observation_low = np.zeros(14)
        self._observation_space = spaces.Box(low=observation_low, high=observation_high)

        self.seed()
        self.viewer = None


        #contruct the model

#         CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
        CURRENT_PATH = os.getcwd()
        print("CURRENT_PATH",CURRENT_PATH)
        # SCRIPTS_PATH = os.path.dirname(CURRENT_PATH)
        # print("SCRIPTS_PATH",SCRIPTS_PATH)
        MODEL_DATA_PATH = os.path.join(CURRENT_PATH, "envs/model_data") 
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

        pure_dynamics_fun = partial(dynamics_fun, ST=ST, idcontact=idcontact, \
                parent=parent, jtype=jtype, jaxis=jaxis, NB=NB, NC=NC, nf=nf, ncp=ncp)

        pure_events_fun = partial(events_fun, ST=ST, idcontact=idcontact, \
                parent=parent, jtype=jtype, jaxis=jaxis, NB=NB, NC=NC, nf=nf, ncp=ncp)

        pure_impulsive_fun =  partial(impulsive_dynamics_fun, ST=ST, idcontact=idcontact, \
            parent=parent, jtype=jtype, jaxis=jaxis, NB=NB, NC=NC, nf=nf, ncp=ncp)


        def _dynamics_step(y0, *args):
            t_eval, sol =  integrate_dynamics(pure_dynamics_fun, y0, t_span, delta_t, pure_events_fun, pure_impulsive_fun, args=args)
            yT = sol[-1, :]
            return yT

        u = jnp.zeros((4,))
        self.pure_args = (Xtree, I, contactpoint, u, a_grav, contact_force_lb, contact_force_ub,  contact_pos_lb, contact_vel_lb, contact_vel_ub, mu)

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

        Xtree, I, contactpoint, u0, a_grav, contact_force_lb, contact_force_ub, contact_pos_lb, contact_vel_lb, contact_vel_ub,mu = self.pure_args
        pure_args = (Xtree, I, contactpoint, u, a_grav, contact_force_lb, contact_force_ub,  contact_pos_lb, contact_vel_lb, contact_vel_ub, mu)
        next_xk = self.dynamics_step(self.xk, *pure_args)
        loss = jnp.sum((q_star[3:7] - xk[3:7])**2) + jnp.sum((qdot_star[3:7] - xk[10:14])**2)
        reward = - loss
        self.xk = next_xk
        self.state = np.array(xk) # update jnp state
        next_state = self.state # convert back to numpy.ndarray
        done = None
        return next_state, reward, done, {}

    def reset(self):
        self.xk = jnp.zeros((14,)) #xk refers to jaxRBDL state
        self.state = np.array(self.xk)
        return self.state
    ...
    # def render(self, mode='human', close=False):`
    # ...

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None