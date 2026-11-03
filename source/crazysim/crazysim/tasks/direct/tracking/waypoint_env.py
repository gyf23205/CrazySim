from __future__ import annotations

import torch

from isaaclab.utils.math import euler_xyz_from_quat

from . import waypoint_control
from .firmware_controller import FirmwarePositionController
from .tracking_env import TrackingEnv
from .waypoint_env_cfg import WaypointTrackingEnvCfg


class WaypointTrackingEnv(TrackingEnv):
    """Crazyflie following a reference with drone-relative waypoints sent to the firmware position controller.

    Simulation of ``examples/trajectory_setpoints`` of ``Drone_Test_TRO``. The policy observes all setpoints of
    one lap relative to the drone and outputs a waypoint relative to the drone. Reward and crash checks follow
    the same repository; see ``waypoint_control.py`` and ``firmware_controller.py``.
    """

    cfg: WaypointTrackingEnvCfg
    ACTION_MODES = ("relative_waypoint",)
    REWARD_TERMS = ("alive", "coarse_position", "precise_position", "velocity", "waypoint_error", "crash", "success")

    @staticmethod
    def _observation_size(cfg: WaypointTrackingEnvCfg) -> int:
        return 3 * cfg.setpoint_count + 8

    def __init__(self, cfg: WaypointTrackingEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        if self.cfg.command_constraint not in ("slew", "proximity"):
            raise ValueError(f"Unknown command_constraint '{self.cfg.command_constraint}'.")

        self._offset_limit = torch.tensor(self.cfg.waypoint_offset_limit, device=self.device)
        self._position_scale = torch.tensor(self.cfg.observation_position_scale, device=self.device)
        self._rate_limit = torch.tensor(self.cfg.command_rate_limit, device=self.device)
        self._proximity_radius = torch.tensor(self.cfg.command_proximity_radius, device=self.device)
        self._position_precision = torch.tensor(self.cfg.position_precision_scale, device=self.device)
        self._velocity_precision = torch.tensor(self.cfg.velocity_precision_scale, device=self.device)

        self._hover_motor_thrust = self._robot_weight / 4.0
        self._firmware = FirmwarePositionController(
            self.num_envs,
            self.device,
            self.physics_dt,
            self.cfg.motor_thrust_max,
            self.cfg.motor_thrust_min,
            self.cfg.motor_time_constant,
            self.cfg.firmware_xy_velocity_max,
            self.cfg.firmware_z_velocity_max,
        )
        self.applied_command = torch.zeros(self.num_envs, 3, device=self.device)
        """Position setpoint sent to the firmware, relative to the environment origin [m]."""
        self._action_position = torch.zeros(self.num_envs, 3, device=self.device)
        self._previous_velocity = torch.zeros(self.num_envs, 3, device=self.device)

        self._tracking_steps = torch.zeros(self.num_envs, device=self.device)
        self._position_error_squared_sum = torch.zeros(self.num_envs, device=self.device)
        self._cross_error_squared_sum = torch.zeros(self.num_envs, device=self.device)
        self._within_distance_sum = torch.zeros(self.num_envs, device=self.device)
        self._completed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.cross_error = torch.zeros(self.num_envs, device=self.device)
        """Distance to the reference across its direction of travel, at the last step [m]."""
        self.episode_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        """True on the step an episode reaches the end of the reference and passes the quality check."""
        self.crash_causes = {}
        """Boolean tensors of the crash checks of the last step."""

    def set_task(self, name: str):
        super().set_task(name)
        self.cfg.episode_length_s = self.cfg.trajectory_laps * self.cfg.tasks[name].period

    def _position(self) -> torch.Tensor:
        return self._robot.data.root_pos_w - self._terrain.env_origins

    def _command_from_action(self, actions: torch.Tensor) -> torch.Tensor:
        """``_map_action_to_absolute_command`` of ``Drone_Test_TRO``."""
        z_min, z_max = self.cfg.command_z_range
        target = waypoint_control.action_to_waypoint(actions, self._action_position, self._offset_limit)
        target = waypoint_control.clamp_to_geofence(target, self.cfg.command_xy_limit, z_min, z_max)
        if self.cfg.command_constraint == "proximity":
            command = waypoint_control.proximity_limit(target, self._action_position, self._proximity_radius)
        else:
            command = waypoint_control.slew_limit(target, self.applied_command, self._rate_limit, self.step_dt)
        return waypoint_control.clamp_to_geofence(command, self.cfg.command_xy_limit, z_min, z_max)

    def _pre_physics_step(self, actions: torch.Tensor):
        self._previous_actions = self._actions.clone()
        # _clip_action of Drone_Test_TRO: an action outside [-1, 1] or not finite holds the last command
        hold_command = (~torch.isfinite(actions) | (actions.abs() > 1.0)).any(dim=1, keepdim=True)
        self._actions = torch.nan_to_num(actions, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)
        self._action_position = self._position().clone()
        self._previous_velocity = self._robot.data.root_lin_vel_w.clone()
        self.applied_command = torch.where(hold_command, self.applied_command, self._command_from_action(self._actions))
        # the reference this step is scored against is the one at the end of the step
        self._trajectory_time += self.step_dt

    def _attitude_deg(self) -> torch.Tensor:
        roll, pitch, yaw = euler_xyz_from_quat(self._robot.data.root_quat_w)
        attitude = torch.stack([roll, pitch, yaw], dim=-1)
        # euler_xyz_from_quat returns angles in [0, 2 pi)
        attitude = torch.remainder(attitude + torch.pi, 2.0 * torch.pi) - torch.pi
        return torch.rad2deg(attitude)

    def _apply_action(self):
        motor_thrust = self._firmware.step(
            self.applied_command,
            self._position(),
            self._robot.data.root_lin_vel_w,
            self._attitude_deg(),
            torch.rad2deg(self._robot.data.root_ang_vel_b),
        )
        self._set_motor_thrusts(motor_thrust)
        self._robot.set_external_force_and_torque(self._thrust, self._moment, body_ids=self._body_id)

    def _get_observations(self) -> dict:
        phase = torch.remainder(self._trajectory_time / self._trajectory.cfg.period, 1.0)
        obs = waypoint_control.build_observation(
            self._position(),
            self._robot.data.root_lin_vel_w,
            self._setpoints,
            phase,
            self.applied_command,
            self._position_scale,
            self.cfg.observation_velocity_scale,
            self._offset_limit,
        )
        return {"policy": obs}

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._update_reference()
        position = self._position()
        velocity = self._robot.data.root_lin_vel_w
        attitude = self._attitude_deg()
        acceleration = (velocity - self._previous_velocity) / self.step_dt / self._gravity_magnitude
        self.crash_causes = {
            "height": (position[:, 2] < self.cfg.min_height) | (position[:, 2] > self.cfg.max_height),
            "horizontal": (position[:, :2].abs() > self.cfg.max_horizontal_distance).any(dim=1),
            "speed": (torch.linalg.norm(velocity[:, :2], dim=1) > self.cfg.speed_limit)
            | (velocity[:, 2].abs() > self.cfg.speed_limit),
            "acceleration": (acceleration.abs() > self.cfg.max_acceleration_g).any(dim=1),
            "tilt": (attitude[:, :2].abs() > self.cfg.max_tilt_deg).any(dim=1),
            "angular_velocity": (
                torch.rad2deg(self._robot.data.root_ang_vel_b).abs() > self.cfg.max_angular_velocity_deg
            ).any(dim=1),
        }
        self._crashed = torch.stack(list(self.crash_causes.values())).any(dim=0)
        self._lost = self.position_error > self.cfg.max_position_error
        # reaching the end of the reference: terminal without bootstrapping (agent.time_limit_bootstrap: False)
        self._completed = (self.episode_length_buf >= self.max_episode_length) & ~self._crashed & ~self._lost
        return self._crashed | self._lost, self._completed

    def _get_rewards(self) -> torch.Tensor:
        position = self._position()
        reference_position = self._reference_pos - self._terrain.env_origins
        desired_action = waypoint_control.waypoint_to_action(
            reference_position, self._action_position, self._offset_limit
        )
        rewards, metrics = waypoint_control.reward_components(
            position,
            self._robot.data.root_lin_vel_w,
            reference_position,
            self._reference_vel,
            self._actions,
            desired_action,
            self.step_dt,
            self._position_precision,
            self._velocity_precision,
            self.cfg.coarse_position_distance,
            self.cfg.velocity_gate_distance,
            self.cfg.alive_reward_scale,
            self.cfg.coarse_position_reward_scale,
            self.cfg.precise_position_reward_scale,
            self.cfg.velocity_reward_scale,
            self.cfg.waypoint_error_reward_scale,
        )
        self.cross_error = metrics["cross_error"]
        self._tracking_steps += 1.0
        self._position_error_squared_sum += torch.square(metrics["position_error"])
        self._cross_error_squared_sum += torch.square(metrics["cross_error"])
        self._within_distance_sum += (metrics["position_error"] <= self.cfg.success_distance_threshold).float()

        passes = (
            torch.sqrt(self._position_error_squared_sum / self._tracking_steps) <= self.cfg.success_position_rmse
        ) & (torch.sqrt(self._cross_error_squared_sum / self._tracking_steps) <= self.cfg.success_cross_track_rmse) & (
            self._within_distance_sum / self._tracking_steps >= self.cfg.success_distance_fraction
        )
        self.episode_success = self._completed & passes
        rewards["crash"] = self.cfg.crash_penalty * self._crashed.float()
        rewards["success"] = self.cfg.success_bonus * self.episode_success.float()

        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        for key, value in rewards.items():
            self._episode_sums[key] += value
        self._position_error_sum += metrics["position_error"]
        return reward

    def _start_times(self, count: int) -> torch.Tensor:
        if self.cfg.random_start_phase:
            return super()._start_times(count)
        return torch.zeros(count, device=self.device)

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES
        super()._reset_idx(env_ids)
        # after the base class has replaced extras["log"] with the statistics of the finished episodes
        self.extras["log"]["Episode_Termination/success"] = torch.count_nonzero(self.episode_success[env_ids]).float()
        position = self._position()[env_ids]
        self._firmware.reset(env_ids, self._hover_motor_thrust, position)
        self.applied_command[env_ids] = self._trajectory.position(self._trajectory_time[env_ids])
        self._action_position[env_ids] = position
        self._previous_velocity[env_ids] = 0.0
        self._tracking_steps[env_ids] = 0.0
        self._position_error_squared_sum[env_ids] = 0.0
        self._cross_error_squared_sum[env_ids] = 0.0
        self._within_distance_sum[env_ids] = 0.0
