from gymnasium.envs.registration import register
import numpy as np
from environment_gym.envs.uav_avoid import UavAvoidEnv

register(
    id='environment_gym/UavAvoid-v0',
    entry_point="environment_gym.envs:UavAvoidEnv",
    kwargs={
        "agent_r": 1.0,
        "action_area": np.array([[0, 0, 0], [100, 100, 25]]),
        "action_bound": 2.0,
        "uavs_num": 15,
        "render_mode": None
    },
)