import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='jbdl_half_cheetah-v0',
    entry_point='gym_rbdl.envs:HalfCheetahRBDLEnv',
    max_episode_steps=1000,
    reward_threshold=8.0,
    nondeterministic = True,
)

# register(
#     id='jbdl_cartpole-v0',
#     entry_point='gym_rbdl.envs:CartpoleRBDLEnv',
#     max_episode_steps=500,
#     reward_threshold=8.0,
#     nondeterministic = True,
# )

register(
    id='jbdl_cartpole-v1',
    entry_point='gym_rbdl.envs:CartPoleJBDLEnv',
    max_episode_steps=500,
    reward_threshold=8.0,
    nondeterministic = True,
)

register(
    id='jbdl_pendulum-v0',
    entry_point='gym_rbdl.envs:PendulumJBDLEnv',
    max_episode_steps=200,
    reward_threshold=8.0,
    nondeterministic = True,
)

register(
    id='jbdl_pendulum-v1',
    entry_point='gym_rbdl.envs:PendulumEnv',
    max_episode_steps=500,
    reward_threshold=8.0,
    nondeterministic = True,
)

register(
    id='jbdl_arm-v0',
    entry_point='gym_rbdl.envs:ArmJBDLEnv',
    max_episode_steps=500,
    reward_threshold=8.0,
    nondeterministic = True,
)