import gym
from gym import error, spaces, utils
from gym.utils import seeding

import math
import gym
import numpy as np

import math
import jax
import jax.numpy as jnp
from jbdl.rbdl.dynamics.forward_dynamics import  forward_dynamics_core
import pybullet as p
import pybullet_data
from jbdl.rbdl.model.rigid_body_inertia import rigid_body_inertia, init_Ic_by_cholesky
from jbdl.rbdl.utils import xyz2int
from functools import partial
from jbdl.experimental.ode.runge_kutta import odeint
from jax import jit, vmap
from jax.ops import index_update, index
# from jbdlenvs.utils.parser import URDFBasedRobot

from Simulator.UrdfWrapper import UrdfWrapper
#   from jaxRBDL.Utils.UrdfWrapper_guo import UrdfWrapper
from Simulator.ObdlRender import ObdlRender
from Simulator.ObdlSim import ObdlSim

M_CART = 1.0
M_POLE = 0.1
HALF_POLE_LENGTH = 0.5
POLE_IC_PARAMS = jnp.zeros((6,))
DEFAULT_PURE_Pendulum_PARAMS = (M_CART, M_POLE, HALF_POLE_LENGTH, POLE_IC_PARAMS)


def init_I(m, c, l):
    Ic =  init_Ic_by_cholesky(l)
    I = rigid_body_inertia(m, c, Ic)

    return I



class PendulumJBDLEnv(gym.Env):
    """
    
    """

    def __init__(self, pure_pendulum_params=DEFAULT_PURE_Pendulum_PARAMS, reward_fun=None, seed=0, batch_size=0, render=False):
        self._init_params(*pure_pendulum_params)
        
        action_max = np.ones(1)*100.
        self._action_space = spaces.Box(low=-action_max, high=action_max)
        observation_high = np.ones(4)*math.pi
        observation_low = np.zeros(4)
        self._observation_space = spaces.Box(low=observation_low, high=observation_high)


        self.NB = 2
        self.nf = 3
        self.a_grav = jnp.array([[0.], [0.], [0.], [0.], [0.], [-9.81]])
        self.jtype = (1, 0)
        self.jaxis = xyz2int('xy')
        self.parent = (0, 1)
        self.Xtree = list([jnp.eye(6) for i in range(self.NB)])
        self.sim_dt = 0.1
        self.batch_size = batch_size
 
        

        self.render = render

        # Angle at which to fail the episode
        # Angle at which to fail the episode
        self.theta_threshold = 15.0 / 360.0 * math.pi
        self.x_threshold = 2.5
        self.key = jax.random.PRNGKey(seed)

        if self.render:
            p.connect(p.GUI)
            p.resetDebugVisualizerCamera(cameraDistance=6.18, cameraYaw=0, cameraPitch=-30, cameraTargetPosition=[0, 0, 1.0])
            self.pendulum_render = URDFBasedRobot("inverted_pendulum.urdf", "physics", action_dim=1, obs_dim=4)
            self.pendulum_render.load(p)
 
        self.model = UrdfWrapper("urdf/inverted pendulum_link1_1.urdf").model
        self.osim = ObdlSim(self.model,dt=0.02,vis=True)

        self.reset()

        def _reset_pendulum_render(bullet_client, pendulum, x, theta):
            pendulum.slider_to_cart = pendulum.jdict["slider_to_cart"]
            pendulum.cart_to_pole = pendulum.jdict["cart_to_pole"]
            pendulum.slider_to_cart.reset_current_position(x, 0)
            pendulum.cart_to_pole.reset_current_position(theta, 0)

        self.reset_pendulum_render = _reset_pendulum_render


        def _dynamics_fun(y, t, Xtree, I, u, a_grav, parent, jtype, jaxis, NB):
            q = y[0:NB]
            qdot = y[NB:]
            input = (Xtree, I, parent, jtype, jaxis, NB, q, qdot, u, a_grav)
            qddot = forward_dynamics_core(*input)
            ydot = jnp.hstack([qdot, qddot])
            return ydot

        self.dynamics_fun = partial(_dynamics_fun, parent=self.parent, jtype=self.jtype, jaxis=self.jaxis, NB=self.NB)
        
        # @partial(jit, static_argnums=0)
        def _dynamics_step(dynamics_fun, y0, *args, sim_dt=self.sim_dt, rtol=1.4e-8, atol=1.4e-8, mxstep=jnp.inf):
            t_eval = jnp.linspace(0, sim_dt, 2)
            y_all = odeint(dynamics_fun, y0, t_eval, *args, rtol=rtol, atol=atol, mxstep=mxstep)
            yT = y_all[-1, :]
            return yT

        self._dynamics_step = _dynamics_step

        self.dynamics_step = jit(_dynamics_step, static_argnums=0)


        def _dynamics_step_with_params(dynamics_fun, state, action, *pendulum_params, Xtree=self.Xtree, a_grav=self.a_grav, sim_dt=self.sim_dt, rtol=1.4e-8, atol=1.4e-8, mxstep=jnp.inf):
            m_cart, m_pole, half_pole_length, pole_Ic_params = pendulum_params
            I_cart = init_I(m_cart, jnp.zeros((3,)), jnp.zeros((6,)))
            I_pole = init_I(m_pole, jnp.array([0.0, 0.0, half_pole_length]), pole_Ic_params)
            I = [I_cart, I_pole]
            u = jnp.array([action[0], 0.0])
            dynamics_fun_param = (Xtree, I, u, a_grav)
            next_state = self._dynamics_step(dynamics_fun, state, *dynamics_fun_param, sim_dt=sim_dt, rtol=rtol, atol=atol, mxstep=mxstep)
            return next_state

        self._dynamics_step_with_params =  _dynamics_step_with_params
        self.dynamics_step_with_params = jit(_dynamics_step_with_params, static_argnums=0)


        def _done_fun(state, x_threshold=self.x_threshold, theta_threshold=self.theta_threshold):
            x = state[0]
            theta = state[1]
            done = jax.lax.cond(
                (jnp.abs(x) > jnp.abs(x_threshold)) + (jnp.abs(theta) > jnp.abs(theta_threshold)),
                lambda done: True,
                lambda done: False,
                None)
            return done
        
        self._done_fun = _done_fun

        self.done_fun = jit(_done_fun)


       
        def _default_reward_fun(state, action, next_state):
            reward = -(next_state[0]**2 + 10 * next_state[1]**2 + next_state[2]**2 + next_state[3]**2)
            return reward

        if reward_fun is None:
            self._reward_fun = _default_reward_fun
            self.reward_fun = jit(_default_reward_fun)
        else:
            self._reward_fun = reward_fun
            self.reward_fun = jit(reward_fun)



    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self._observation_space

        

            

    def _init_params(self, *cart_pole_params):
        self.m_cart, self.m_pole, self.half_pole_length, self.pole_Ic_params = cart_pole_params
        self.I_cart = init_I(self.m_cart, jnp.zeros((3,)), jnp.zeros((6,)))
        self.I_pole = init_I(self.m_pole, jnp.array([0.0, 0.0, self.half_pole_length]), self.pole_Ic_params)
        self.I = [self.I_cart, self.I_pole]


    def reset(self, m_cart=M_CART, m_pole=M_POLE, half_pole_length=HALF_POLE_LENGTH, pole_Ic_params=POLE_IC_PARAMS, idx_list=None):
        self._init_params(m_cart, m_pole, half_pole_length, pole_Ic_params)
        self.key, subkey = jax.random.split(self.key)
        if self.batch_size == 0:
            self.state = jax.numpy.array(
                [0, jax.random.uniform(key=self.key, shape=(),
                                       minval=-15.0/360.0*math.pi,
                                       maxval=15.0/360.0*math.pi), 0, 0])
        else:
            if idx_list is None:
                self.state = jax.numpy.concatenate([
                        jax.numpy.zeros(shape=(self.batch_size, 1)),
                        jax.random.uniform(
                            self.key, shape=(self.batch_size, 1),
                            minval=-15.0/360.0*math.pi,
                            maxval=15.0/360.0*math.pi),
                        jax.numpy.zeros(shape=(self.batch_size, 1)),
                        jax.numpy.zeros(shape=(self.batch_size, 1))], axis=-1)
            else:
                idx_num = len(idx_list)
                self.state = index_update(
                    self.state,
                    index[idx_list, :],
                    jax.numpy.concatenate([
                        jax.numpy.zeros(shape=(idx_num, 1)),
                        jax.random.uniform(
                            self.key, shape=(idx_num, 1),
                            minval=-15.0/360.0*math.pi,
                            maxval=15.0/360.0*math.pi),
                        jax.numpy.zeros(shape=(idx_num, 1)),
                        jax.numpy.zeros(shape=(idx_num, 1))], axis=-1)
                )
        self.state =  np.array(self.state)
        return self.state



    def step(self, action):

        if self.batch_size == 0:
            u = jnp.array([action[0], 0.0])
            dynamics_params = (self.Xtree, self.I, u, self.a_grav)
            next_state = self.dynamics_step(self.dynamics_fun, self.state, *dynamics_params)
            done = self.done_fun(next_state)
            reward = self.reward_fun(self.state, action, next_state)
            self.state = next_state
        else:
            action = jnp.reshape(jnp.array(action), newshape=(self.batch_size, 1))
            u = jnp.concatenate([action, jnp.zeros((self.batch_size, 1))], axis=1)
            dynamics_params = (self.Xtree, self.I, u, self.a_grav)
            next_state = vmap(self.dynamics_step, (None, 0, None, None, 0, None), 0)(self.dynamics_fun, self.state, *dynamics_params)
            done = vmap(self.done_fun)(next_state)
            reward = vmap(self.reward_fun)(self.state, action, next_state)
            self.state = next_state
        next_state =  np.array(next_state)
        reward =  np.array(reward)
        return next_state, reward, done, {}

    def pb_render(self, idx=0):
        if self.render:
            if self.batch_size == 0:
                self.reset_pendulum_render(p, self.pendulum_render, self.state[0], self.state[1])
            else:
                self.reset_pendulum_render(p, self.pendulum_render, self.state[idx, 0], self.state[idx, 1])

    def osim_render(self):
        q = [0,0,self.state[0],self.state[1]]
        self.osim.step_theta(q)
