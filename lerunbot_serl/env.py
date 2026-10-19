import lerunbot  # Don't remove this import. In its __init__.py, we register the Lerunbot env

from functools import partial
from torchrl.envs import GymWrapper, EnvCreator, ParallelEnv
from torchrl.envs.transforms import (
    TransformedEnv,
    Compose,
    InitTracker,
    StepCounter,
    RewardSum,
    DoubleToFloat,
)
import gymnasium as gym
from omegaconf import OmegaConf
import gym_lowcostrobot  # Don't remove this import. In its __init__.py, we register the LowCostRobot envs

# def DiscountedRewardSum(gamma, data):
#     if data["is_init"]:
#         data["discounted_reward_sum"] = 0.0
#         data["discount"] = 1.0
#     if "next" in data.keys():
#         data["next", "discounted_reward_sum"] = (
#             data["discounted_reward_sum"] + data["discount"] * data["next", "reward"]
#         )
#         data["next", "discount"] = data["discount"] * gamma
#     return data


# Need to ensure the returned env has correct specs
def make_single_environment(cfg, eval=False):
    if cfg.env.type == "gym":
        env = gym.make(cfg.env.gym.name, **OmegaConf.to_container(cfg.env.gym[cfg.env.gym.name].args, resolve=True))
        env = GymWrapper(env, **OmegaConf.to_container(cfg.env.gym.args, resolve=True))
        if eval:
            env.auto_register_info_dict()
        # env = GymEnv(cfg.env.gym.name, from_pixels=from_pixels, pixels_only=False)
    else:
        raise NotImplementedError(f"Environment type {cfg.env.type} not implemented")
    env = TransformedEnv(env, DoubleToFloat())
    env.set_seed(cfg.experiment.seed)
    return env


def make_environment(cfg, eval=False):
    env = ParallelEnv(
        create_env_fn=EnvCreator(partial(make_single_environment, cfg=cfg, eval=eval)),
        serial_for_single=True,  # creating a parallel environment with a single worker will return a SerialEnv instead, avoid creating parallel envs if not necessary. All the interfaces will remain the same.
        **OmegaConf.to_container(cfg.env.parallel_env_args, resolve=True),
    )
    env.set_seed(cfg.experiment.seed)
    env = TransformedEnv(
        env,
        Compose(
            InitTracker(),
            StepCounter(cfg.env.max_steps),
            DoubleToFloat(),
            # partial(DiscountedRewardSum, cfg.env.gamma),
            RewardSum(),
        ),
    )
    return env
