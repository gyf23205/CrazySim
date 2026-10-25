"""Crazyflie trajectory-tracking environment.

A single gym id covers every tracking task. The task (circle, rectangle, figure8, star, vertical_circle) is selected
with ``env.unwrapped.set_task(name)``, so a sequence of tasks can be trained in one process.
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="CrazySim-Tracking-Direct-v0",
    entry_point=f"{__name__}.tracking_env:TrackingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tracking_env_cfg:TrackingEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)
