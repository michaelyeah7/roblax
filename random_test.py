from utils import Random
import jax
from jax import random
import jax.numpy as jnp
# random = Random(0)
# state = jax.random.uniform(
#     random.get_key(), shape=(4,), minval=-0.05, maxval=0.05
# )
# print("state",state)

# x  = random.randint(random.PRNGKey(1), (10,), 100, 254, dtype='uint8')
# print("x",x)


# x =  jnp.ones(4)
# x = jnp.vstack((x, jnp.ones(4)))
# x = jnp.vstack((x, jnp.ones(4)))
x = jnp.array([[1,1,1,1]])
y = x.mean()
print("x",y)