import gym
from gym import error, spaces, utils
from gym.utils import seeding
import math
import numpy as np
import jax.numpy as jnp

class HalfCheetahRBDLEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        action_max = np.ones(4)*10.
        self._action_space = spaces.Box(low=-action_max, high=action_max)
        observation_high = np.ones(7)*math.pi
        observation_low = np.zeros(7)
        self._observation_space = spaces.Box(low=observation_low, high=observation_high)

        self.seed()
        self.viewer = None

    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self._observation_space

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _dynamics_step(self, y0, *args):
        t_eval, sol =  integrate_dynamics(pure_dynamics_fun, y0, t_span, delta_t, pure_events_fun, pure_impulsive_fun, args=args)
        yT = sol[-1, :]
        return yT

    def step(self, action):
        q_star = jnp.array([0.0,  0.0, 0.0, math.pi/6, -math.pi/6, -math.pi/3, math.pi/3])
        qdot_star = jnp.zeros((7, ))
        u = jnp.array(action)  # convert numpy.ndarray to jnp
        pure_args = (Xtree, I, contactpoint, u, a_grav, contact_force_lb, contact_force_ub,  contact_pos_lb, contact_vel_lb, contact_vel_ub, mu)
        xk = self._dynamics_step(self.state, *pure_args)
        loss = jnp.sum((q_star[3:7] - xk[3:7])**2) + jnp.sum((qdot_star[3:7] - xk[10:14])**2)
        reward = - loss
        self.state = xk # update jnp state
        next_state = np.array(xk) # convert back to numpy.ndarray
        done = None
        return next_state, reward, done, {}

    def reset(self):
        self.state = jnp.zeros((14,))
    ...
    # def render(self, mode='human', close=False):`
    # ...

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None