from os import path

import gym
import jax
import jax.numpy as jnp
import numpy as np

# from envs.core import Env
# from utils import Random


@jax.jit
def angle_normalize(x):
    return ((x + jnp.pi) % (2 * jnp.pi)) - jnp.pi


def default_reward_fn(x, u):
    return -(np.sum(angle_normalize(x[0]) ** 2 + 0.1 * x[1] ** 2 + 0.001 * (u ** 2)))


class Pendulum():
    max_speed = 8.0
    max_torque = 2.0  # gym 2.
    high = np.array([1.0, 1.0, max_speed])

    action_space = gym.spaces.Box(low=-max_torque, high=max_torque, shape=(1,), dtype=np.float32)
    observation_space = gym.spaces.Box(low=-high, high=high, dtype=np.float32)

    def __init__(self, render_flag=False, reward_fn=None, seed=0, horizon=50):
        # self.reward_fn = reward_fn or default_reward_fn
        self.dt = 0.05
        self.viewer = None

        self.state_size = 2
        self.action_size = 1
        self.action_dim = 1 # redundant with action_size but needed by ILQR
        
        self.H = horizon

        self.n, self.m = 2, 1
        self.angle_normalize = angle_normalize
        self.nsamples = 0

        # self.random = Random(seed)

        self.render_flag = render_flag

        self.reset()
        
        # @jax.jit
        def _dynamics(state, action):
            self.nsamples += 1
            th, thdot = state
            g = 10.0
            m = 1.0
            ell = 1.0
            dt = self.dt

            # Do not limit the control signals
            # action = jnp.clip(action, -self.max_torque, self.max_torque)
            action = action

            newthdot = (
                thdot + (-3 * g / (2 * ell) * jnp.sin(th + jnp.pi) + 3.0 / (m * ell ** 2) * action) * dt
            )
            newth = th + newthdot * dt
            # newthdot = jnp.clip(newthdot, -self.max_speed, self.max_speed)
            newthdot = newthdot

            return jnp.reshape(jnp.array([newth, newthdot]), (2,))
        
        @jax.jit
        def c(x, u):
            # return np.sum(angle_normalize(x[0]) ** 2 + 0.1 * x[1] ** 2 + 0.001 * (u ** 2))
            return angle_normalize(x[0])**2 + .1*(u[0]**2)
        
        self.reward_fn = reward_fn or c
        self.dynamics = _dynamics
        self.f, self.f_x, self.f_u = (
                _dynamics,
                jax.jacfwd(_dynamics, argnums=0),
                jax.jacfwd(_dynamics, argnums=1),
            )
        self.c, self.c_x, self.c_u, self.c_xx, self.c_uu = (
                c,
                jax.grad(c, argnums=0),
                jax.grad(c, argnums=1),
                jax.hessian(c, argnums=0),
                jax.hessian(c, argnums=1),
            )


    def step(self, state, action):

        self.state = self.dynamics(state, action)

        
        reward = self.reward_func(self.state)
        return self.state, reward, False, {}

    def reward_func(self,state):
        th, thdot =  state
        # costs = angle_normalize(th) ** 2 + .1 * thdot ** 2 + .001 * (u ** 2)
        
        # costs = angle_normalize(th) ** 2 + .1 * thdot ** 2 
        return -(jnp.sum(angle_normalize(th) ** 2 + 0.1 * thdot ** 2))

    def reset(self):
        # th = jax.random.uniform(self.random.generate_key(), minval=-jnp.pi, maxval=jnp.pi)
        # thdot = jax.random.uniform(self.random.generate_key(), minval=-1.0, maxval=1.0)

        th = float(np.random.uniform(-np.pi,np.pi,1))
        thdot = float(np.random.uniform(-1.0,1.0,1))

        self.state = jnp.array([th, thdot])

        return self.state

    def render(self, mode="human"):
        if self.viewer is None:
            from gym.envs.classic_control import rendering

            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-2.2, 2.2, -2.2, 2.2)
            rod = rendering.make_capsule(1, 0.2)
            rod.set_color(0.8, 0.3, 0.3)
            self.pole_transform = rendering.Transform()
            rod.add_attr(self.pole_transform)
            self.viewer.add_geom(rod)
            axle = rendering.make_circle(0.05)
            axle.set_color(0, 0, 0)
            self.viewer.add_geom(axle)
            fname = path.join(path.dirname(__file__), "assets/clockwise.png")
            self.img = rendering.Image(fname, 1.0, 1.0)
            self.imgtrans = rendering.Transform()
            self.img.add_attr(self.imgtrans)

        self.viewer.add_onetime(self.img)
        self.pole_transform.set_rotation(self.state[0] + np.pi / 2)
        # if self.last_u:
        #     self.imgtrans.scale = (-self.last_u / 2, np.abs(self.last_u) / 2)

        return self.viewer.render(return_rgb_array=mode == "rgb_array")