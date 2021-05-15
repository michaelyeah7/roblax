import jax.numpy as jnp
import numpy as np

# x = jnp.array([-1000.0, -1000.0, 0.]).reshape(-1, 1)
# x = x.flatten()

# # z =  np.array([1,2,3])
# y = x[jnp.array([0, 2])]
# print("y shape",y)

L = []
L = np.vstack([L, -np.eye(4)])
print("L",L)