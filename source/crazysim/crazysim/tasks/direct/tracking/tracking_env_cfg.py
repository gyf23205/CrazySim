from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from .trajectories import CircleCfg, Figure8Cfg, RectangleCfg, TrajectoryCfg

##
# Pre-defined configs
##
from isaaclab_assets import CRAZYFLIE_CFG  # isort: skip


@configclass
class TrackingEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 10.0
    decimation = 2
    action_space = 4
    # observation: reference (see observation_mode), linear velocity (3), angular velocity (3),
    # rotation matrix (9), previous action (4)
    observation_mode: str = "setpoints"
    """How the reference is observed, always relative to the robot position:

    - ``"lookahead"``: reference points at the current reference time plus ``lookahead_times``
      (``3 * len(lookahead_times)`` values). They move with the reference.
    - ``"setpoints"``: ``setpoint_count`` setpoints spread evenly in time over one whole traversal,
      fixed in the world, plus the current phase of the traversal as sin and cos
      (``3 * setpoint_count + 2`` values). The policy has to find the current target from the shape.
    """
    lookahead_times: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
    """Times ahead of the current reference time at which reference points are observed [s]."""
    setpoint_count: int = 16
    """Number of setpoints in the ``"setpoints"`` observation mode."""
    observation_space = 3 * setpoint_count + 2 + 3 + 3 + 9 + 4
    state_space = 0
    debug_vis = True

    # simulation (100 Hz control)
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=2.5, replicate_physics=True, clone_in_fabric=True
    )

    # robot
    robot: ArticulationCfg = CRAZYFLIE_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # actuation: the policy outputs one thrust per motor. Motor order and the mixing constants
    # follow the Crazyflie 2.x firmware (power_distribution_quadrotor.c):
    # m1 front-right, m2 back-right, m3 back-left, m4 front-left.
    thrust_to_weight = 1.9
    """Maximum total thrust of the four motors divided by the robot weight."""
    arm_length = 0.046
    """Distance from the center of mass to each motor [m]."""
    thrust_to_torque = 0.005964552
    """Yaw reaction torque per unit motor thrust [m]."""

    # tasks
    task: str = "circle"
    """Task active at start-up. Switch at runtime with ``TrackingEnv.set_task``."""
    tasks: dict[str, TrajectoryCfg] = {
        "circle": CircleCfg(),
        "rectangle": RectangleCfg(),
        "figure8": Figure8Cfg(),
    }

    # reset
    init_pos_noise = 0.1
    """Half-range of the uniform noise added to the spawn position around the reference [m]."""

    # termination
    min_height = 0.1
    max_height = 2.5
    max_position_error = 1.0
    """Episode ends when the robot is farther than this from the reference [m]."""

    # reward scales
    position_tracking_reward_scale = 10.0
    position_tracking_std = 0.25
    velocity_tracking_reward_scale = -0.5
    ang_vel_reward_scale = -0.05
    yaw_reward_scale = -1.0
    action_rate_reward_scale = -0.1
