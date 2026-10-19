import os

from gymnasium.envs.registration import register

__version__ = "0.0.1"

register(
    id="Lerunbot-v0",
    entry_point="lerunbot.envs.lerunbot_env:LerunbotEnv",
    max_episode_steps=200,
)
