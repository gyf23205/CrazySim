"""Configuration of the trajectory-setpoint controller of ``Drone_Test_TRO`` in simulation.

Every value is a default of ``examples/setpoint/cf_setpoint_ppo_real_common.py`` or
``examples/trajectory_setpoints/observation.py`` there, unless a comment says otherwise. The differences are
listed in ``docs/waypoint_control_differences.md``.
"""

from __future__ import annotations

import math

from isaaclab.utils import configclass

from .tracking_env_cfg import TrackingEnvCfg
from .trajectories import ArcLengthFigure8Cfg, CircleCfg, PolygonCfg, TrajectoryCfg

TRAJECTORY_SPEED = 0.25
"""Speed along the reference curve [m/s] (``DEFAULT_TRAJECTORY_SPEED``)."""
FLIGHT_HEIGHT = 0.75
"""Height of the reference curve [m] (``DEFAULT_TARGET_HEIGHT``)."""
CENTER = (0.0, 0.0, FLIGHT_HEIGHT)


def _figure8_period(a: float, b: float, speed: float, samples: int = 4096) -> float:
    """Midpoint rule over the curve length, as ``figure8_path_length``."""
    d_theta = 2.0 * math.pi / samples
    length = 0.0
    for index in range(samples):
        theta = (index + 0.5) * d_theta
        length += math.hypot(a * math.cos(theta), 2.0 * b * math.cos(2.0 * theta)) * d_theta
    return length / speed


_TRIANGLE_HEIGHT = 0.5 * math.sqrt(3.0)


@configclass
class WaypointTrackingEnvCfg(TrackingEnvCfg):
    # 50 Hz policy (DEFAULT_CONTROL_PERIOD_S = 0.02 s); 500 Hz physics, the rate of the firmware attitude loop
    decimation = 10
    action_space = 3
    action_mode: str = "relative_waypoint"
    """``waypoint = drone position + action * waypoint_offset_limit``, sent to the firmware position controller."""
    observation_mode: str = "trajectory_setpoints"
    setpoint_count: int = 16
    observation_space = 3 * setpoint_count + 8

    waypoint_offset_limit: tuple[float, float, float] = (0.5, 0.5, 0.25)
    """Per-axis waypoint offset from the drone at action +-1 [m]."""
    observation_position_scale: tuple[float, float, float] = (2.0, 2.0, 1.0)
    observation_velocity_scale: float = 0.5

    # command constraints
    command_constraint: str = "slew"
    """``"slew"``: limit how fast the applied setpoint moves. ``"proximity"``: confine it to a box around the drone."""
    command_rate_limit: tuple[float, float, float] = (0.5, 0.5, 0.25)
    """Per-axis speed of the applied setpoint in ``"slew"`` mode [m/s]."""
    command_proximity_radius: tuple[float, float, float] = (0.25, 0.25, 0.125)
    """Per-axis half-width of the box in ``"proximity"`` mode [m]."""
    command_xy_limit: float = 4.5
    """``DEFAULT_XY_GEOFENCE_LIMIT`` [m]."""
    command_z_range: tuple[float, float] = (0.0, 9.85)
    """Padded z bounds of the world (expanse 5 m, padding 0.15 m) [m]."""

    # firmware position controller (see firmware_controller.py)
    motor_thrust_max: float = 0.12
    """``THRUST_MAX`` per motor, Crazyflie 2.1+ propellers [N]."""
    motor_thrust_min: float = 0.012817578393224994
    """``THRUST_MIN`` per motor [N]."""
    thrust_to_torque = 0.0069928948992470565
    """``THRUST2TORQUE``, Crazyflie 2.1+ propellers [m]."""
    motor_time_constant: float = 0.072
    """First-order motor lag [s]. Not in Drone_Test_TRO (the drone has real rotors); identified in arXiv:2404.07837."""
    firmware_xy_velocity_max: float = 1.0
    """``posCtlPid.xVelMax``, ``yVelMax``: firmware default, qfly does not write them [m/s]."""
    firmware_z_velocity_max: float = 1.5
    """``posCtlPid.zVelMax``: ``qfly.QualisysCrazyflie.set_speed_limit`` writes the world speed limit [m/s]."""

    # tasks: default shapes of Drone_Test_TRO; the period follows from the speed
    task: str = "square"
    tasks: dict[str, TrajectoryCfg] = {
        "square": PolygonCfg(
            vertices=((-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)),
            center=CENTER,
            period=4.0 / TRAJECTORY_SPEED,
        ),
        "circle": CircleCfg(radius=0.75, center=CENTER, period=2.0 * math.pi * 0.75 / TRAJECTORY_SPEED),
        "triangle": PolygonCfg(
            vertices=((-0.5, -_TRIANGLE_HEIGHT / 3.0), (0.5, -_TRIANGLE_HEIGHT / 3.0), (0.0, 2.0 * _TRIANGLE_HEIGHT / 3.0)),
            center=CENTER,
            period=3.0 / TRAJECTORY_SPEED,
        ),
        "figure8": ArcLengthFigure8Cfg(
            size=(1.0, 0.5), center=CENTER, period=_figure8_period(1.0, 0.5, TRAJECTORY_SPEED)
        ),
    }
    trajectory_laps: int = 2
    """One episode is this many traversals of the reference; ``episode_length_s`` follows from the active task."""

    # reset
    init_pos_noise = 0.1
    """Half-range of the uniform spawn position noise [m]; the real drone starts within 0.2 m of the reference."""
    random_start_phase: bool = True
    """Start each episode at a random point of the reference. The real drone always starts at the first point."""

    # termination (crash)
    min_height = 0.1
    max_height = 2.0
    max_horizontal_distance: float = 4.5
    max_position_error = 1.0
    """Episode ends, without a penalty, when the robot is farther than this from the reference [m]. Drone_Test_TRO
    has no such check. Only for training: without it, early policies spend whole episodes far from the reference
    (scripts/skrl/validate_waypoint.py turns it off)."""
    speed_limit: float = 1.5
    """Limit on the horizontal speed and on the absolute vertical speed [m/s]."""
    max_acceleration_g: float = 1.0
    """Limit on each absolute world acceleration without gravity [g]."""
    max_tilt_deg: float = 25.0
    """Limit on the absolute roll and pitch [deg]."""
    max_angular_velocity_deg: float = 200.0
    """Limit on each absolute body rate [deg/s]."""

    # reward: per second of tracking, except the crash penalty and the success bonus
    alive_reward_scale: float = 1.0
    coarse_position_reward_scale: float = 2.0
    precise_position_reward_scale: float = 8.0
    velocity_reward_scale: float = 2.0
    waypoint_error_reward_scale: float = -1.0
    crash_penalty: float = -100.0
    success_bonus: float = 10.0
    position_precision_scale: tuple[float, float, float] = (0.20, 0.20, 0.12)
    velocity_precision_scale: tuple[float, float, float] = (0.35, 0.35, 0.20)
    coarse_position_distance: float = 2.0
    velocity_gate_distance: float = 0.40
    success_position_rmse: float = 0.20
    success_cross_track_rmse: float = 0.15
    success_distance_threshold: float = 0.25
    success_distance_fraction: float = 0.90

    def __post_init__(self):
        self.sim.dt = 1.0 / 500.0
        self.sim.render_interval = self.decimation
