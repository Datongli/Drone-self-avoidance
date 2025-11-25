from gymnasium.envs.registration import register
from .environment.staticEnvironment import UavAvoidEnv

register(
    id='UavAvoid-v0',
    entry_point="environment_gym_refactor.environment.staticEnvironment:UavAvoidEnv",
)

__all__ = ["UavAvoidEnv"]