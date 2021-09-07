# %%
from jbdl.envs.simplified_cart_pole_env import SimplifiedCartPole
import jax
import jax.numpy as jnp
import time


env = SimplifiedCartPole()

m_cart = 1.0
m_pole = 0.1
half_pole_length = 0.5
pole_Ic_params = jnp.zeros((6,))
state = jnp.array([ 0.04469439, -2.4385397, 0.07498348, -0.36299074] )
action = jnp.array([0.0, ])
# action_add = jnp.array([1.1, ])


# %%
start = time.time()
next_state = env.dynamics_step_with_params(env.dynamics_fun, state, action, m_cart, m_pole, half_pole_length, pole_Ic_params)
print(next_state)
duration = time.time() - start
print("duration:", duration)

start = time.time()
next_state = env.dynamics_step_with_params(env.dynamics_fun, state, action, m_cart, m_pole, half_pole_length, pole_Ic_params)
print(next_state)
duration = time.time() - start
print("duration:", duration)
print("========================")


# %%
batch_size = 1000
v_state = jnp.repeat(jnp.expand_dims(state, 0), batch_size, axis=0)
v_action = jnp.repeat(jnp.expand_dims(action, 0), batch_size, axis=0)

start = time.time()
v_dynamics_step_with_params = jax.vmap(env.dynamics_step_with_params, (None, 0, 0, None, None, None, None,), 0)
v_next_state = v_dynamics_step_with_params(env.dynamics_fun, v_state, v_action, m_cart, m_pole, half_pole_length, pole_Ic_params)
print(v_next_state)
duration = time.time() - start
print("duration:", duration)

start = time.time()
v_next_state = v_dynamics_step_with_params(env.dynamics_fun, v_state, v_action, m_cart, m_pole, half_pole_length, pole_Ic_params)
print(v_next_state)
duration = time.time() - start
print("duration:", duration)
print("==================")


# %%
start = time.time()
dns_to_action = jax.jit(jax.jacrev(env.dynamics_step_with_params, argnums=[3, 4, 5]), static_argnums=0)
print(dns_to_action(env.dynamics_fun, state, action, m_cart, m_pole, half_pole_length, pole_Ic_params))
duration = time.time() - start
print("duration:", duration)


start = time.time()
print(dns_to_action(env.dynamics_fun, state, action, m_cart, m_pole, half_pole_length, pole_Ic_params))
duration = time.time() - start
print("duration:", duration)

print("==================")

# %%

start = time.time()
v_dns_to_action = jax.vmap(dns_to_action, (None, 0, 0, None, None, None, None,), 0)
print(v_dns_to_action(env.dynamics_fun, v_state, v_action, m_cart, m_pole, half_pole_length, pole_Ic_params))
duration = time.time() - start
print("duration:", duration)


start = time.time()
print(v_dns_to_action(env.dynamics_fun, v_state, v_action, m_cart, m_pole, half_pole_length, pole_Ic_params))
duration = time.time() - start
print("duration:", duration)






# %%
