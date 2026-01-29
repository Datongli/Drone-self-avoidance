from gymnasium.envs.registration import register
from .environment.staticEnvironment import UavAvoidEnvSAC, UavAvoidEnvDDPG


register(
    id='UavAvoid-SAC',
    entry_point="environment_gym_refactor.environment.staticEnvironment:UavAvoidEnvSAC",
)

register(
    id='UavAvoid-DDPG',
    entry_point="environment_gym_refactor.environment.staticEnvironment:UavAvoidEnvDDPG",
)

__all__ = ["UavAvoidEnvSAC", "UavAvoidEnvDDPG"]