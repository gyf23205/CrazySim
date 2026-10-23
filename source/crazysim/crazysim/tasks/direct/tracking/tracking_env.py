from __future__ import annotations

import math
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.math import matrix_from_quat

from .tracking_env_cfg import TrackingEnvCfg
from .trajectories import Trajectory

NUM_PATH_MARKERS = 64
"""Number of markers drawn along each reference path when debug visualization is on."""


def reference_observation_size(cfg: TrackingEnvCfg) -> int:
    """Number of observation values describing the reference (see ``TrackingEnvCfg.observation_mode``)."""
    if cfg.observation_mode == "lookahead":
        return 3 * len(cfg.lookahead_times)
    if cfg.observation_mode == "setpoints":
        return 3 * cfg.setpoint_count + 2
    raise ValueError(f"Unknown observation_mode '{cfg.observation_mode}'. Use 'lookahead' or 'setpoints'.")


class TrackingEnv(DirectRLEnv):
    """Crazyflie following a reference trajectory with direct per-motor thrust commands.

    Every task shares the same observation and action spaces, so one policy can be trained on a
    sequence of tasks. Call :meth:`set_task` and then reset the environment to switch tasks.
    """

    cfg: TrackingEnvCfg

    def __init__(self, cfg: TrackingEnvCfg, render_mode: str | None = None, **kwargs):
        # keep the observation size consistent with the reference settings after CLI overrides
        cfg.observation_space = reference_observation_size(cfg) + 3 + 3 + 9 + 4
        super().__init__(cfg, render_mode, **kwargs)

        # Actions: one normalized thrust per motor in [-1, 1]
        self._actions = torch.zeros(self.num_envs, 4, device=self.device)
        self._previous_actions = torch.zeros(self.num_envs, 4, device=self.device)
        # Total thrust and moment applied to the base of the quadcopter
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)

        # Get specific body indices
        self._body_id = self._robot.find_bodies("body")[0]
        self._robot_mass = self._robot.root_physx_view.get_masses()[0].sum()
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
        self._robot_weight = (self._robot_mass * self._gravity_magnitude).item()
        self._max_motor_thrust = self.cfg.thrust_to_weight * self._robot_weight / 4.0

        # Mixer: body torque = mixer @ motor thrusts (firmware motor order m1..m4)
        arm = self.cfg.arm_length / math.sqrt(2.0)
        kappa = self.cfg.thrust_to_torque
        self._mixer = torch.tensor(
            [
                [-arm, -arm, arm, arm],
                [-arm, arm, arm, -arm],
                [-kappa, kappa, -kappa, kappa],
            ],
            device=self.device,
        )

        # Reference trajectory
        self._lookahead_times = torch.tensor(self.cfg.lookahead_times, device=self.device)
        self._trajectory_time = torch.zeros(self.num_envs, device=self.device)
        self._reference_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self._reference_vel = torch.zeros(self.num_envs, 3, device=self.device)
        self.position_error = torch.zeros(self.num_envs, device=self.device)
        self.set_task(self.cfg.task)

        # When True, a full reset spreads episode start times to avoid synchronized resets.
        # Evaluation turns it off so every episode runs the full length.
        self.spread_episode_starts = True

        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "position_tracking",
                "velocity_tracking",
                "ang_vel",
                "yaw",
                "action_rate",
            ]
        }
        self._position_error_sum = torch.zeros(self.num_envs, device=self.device)

        # add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        self.set_debug_vis(self.cfg.debug_vis)

    @property
    def task_names(self) -> list[str]:
        return list(self.cfg.tasks.keys())

    def set_task(self, name: str):
        """Switch the reference trajectory. Reset the environment afterwards."""
        if name not in self.cfg.tasks:
            raise KeyError(f"Unknown task '{name}'. Available tasks: {self.task_names}")
        self.task_name = name
        trajectory_cfg = self.cfg.tasks[name]
        self._trajectory: Trajectory = trajectory_cfg.class_type(trajectory_cfg)
        # fixed setpoints spread evenly in time over one traversal, relative to the env origin
        times = torch.linspace(0.0, trajectory_cfg.period, self.cfg.setpoint_count + 1, device=self.device)[:-1]
        self._setpoints = self._trajectory.position(times)
        self._path_needs_update = True

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self._previous_actions = self._actions.clone()
        self._actions = actions.clone().clamp(-1.0, 1.0)
        motor_thrust = self._max_motor_thrust * (self._actions + 1.0) / 2.0
        self._thrust[:, 0, 2] = motor_thrust.sum(dim=1)
        self._moment[:, 0, :] = motor_thrust @ self._mixer.T
        # the reference this step is scored against is the one at the end of the step
        self._trajectory_time += self.step_dt

    def _apply_action(self):
        self._robot.set_external_force_and_torque(self._thrust, self._moment, body_ids=self._body_id)

    def _get_observations(self) -> dict:
        robot_pos = self._robot.data.root_pos_w - self._terrain.env_origins
        if self.cfg.observation_mode == "lookahead":
            future_times = self._trajectory_time.unsqueeze(1) + self._lookahead_times
            reference = (self._trajectory.position(future_times) - robot_pos.unsqueeze(1)).flatten(1)
        else:
            phase = 2.0 * math.pi * self._trajectory_time / self._trajectory.cfg.period
            reference = torch.cat(
                [
                    (self._setpoints.unsqueeze(0) - robot_pos.unsqueeze(1)).flatten(1),
                    torch.sin(phase).unsqueeze(1),
                    torch.cos(phase).unsqueeze(1),
                ],
                dim=-1,
            )
        rotation = matrix_from_quat(self._robot.data.root_quat_w)
        obs = torch.cat(
            [
                reference,
                self._robot.data.root_lin_vel_w,
                self._robot.data.root_ang_vel_b,
                rotation.flatten(1),
                self._actions,
            ],
            dim=-1,
        )
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        velocity_error = torch.sum(torch.square(self._robot.data.root_lin_vel_w - self._reference_vel), dim=1)
        ang_vel = torch.sum(torch.square(self._robot.data.root_ang_vel_b), dim=1)
        rotation = matrix_from_quat(self._robot.data.root_quat_w)
        yaw = torch.atan2(rotation[:, 1, 0], rotation[:, 0, 0])
        action_rate = torch.sum(torch.square(self._actions - self._previous_actions), dim=1)
        position_tracking = 1.0 - torch.tanh(self.position_error / self.cfg.position_tracking_std)
        rewards = {
            "position_tracking": position_tracking * self.cfg.position_tracking_reward_scale * self.step_dt,
            "velocity_tracking": velocity_error * self.cfg.velocity_tracking_reward_scale * self.step_dt,
            "ang_vel": ang_vel * self.cfg.ang_vel_reward_scale * self.step_dt,
            "yaw": torch.square(yaw) * self.cfg.yaw_reward_scale * self.step_dt,
            "action_rate": action_rate * self.cfg.action_rate_reward_scale * self.step_dt,
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value
        self._position_error_sum += self.position_error
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # called first after the physics step, so the tracking error used by the rewards is updated here
        self._update_reference()
        height = self._robot.data.root_pos_w[:, 2]
        crashed = (height < self.cfg.min_height) | (height > self.cfg.max_height)
        flipped = self._robot.data.projected_gravity_b[:, 2] > 0.0
        lost = self.position_error > self.cfg.max_position_error
        self._crashed = crashed | flipped
        self._lost = lost
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return self._crashed | lost, time_out

    def _update_reference(self):
        self._reference_pos = self._trajectory.position(self._trajectory_time) + self._terrain.env_origins
        self._reference_vel = self._trajectory.velocity(self._trajectory_time)
        self.position_error = torch.linalg.norm(self._reference_pos - self._robot.data.root_pos_w, dim=1)

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        # Logging
        episode_length = self.episode_length_buf[env_ids].clamp(min=1).float()
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        extras["Metrics/position_error"] = torch.mean(self._position_error_sum[env_ids] / episode_length)
        self._position_error_sum[env_ids] = 0.0
        # tensors (not Python numbers) so that the skrl trainer records them
        if hasattr(self, "_crashed"):
            extras["Episode_Termination/crashed"] = torch.count_nonzero(self._crashed[env_ids]).float()
            extras["Episode_Termination/lost"] = torch.count_nonzero(self._lost[env_ids]).float()
            extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).float()
        self.extras["log"] = extras

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs and self.spread_episode_starts:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        # Start each episode at a random point of the reference
        self._trajectory_time[env_ids] = torch.rand(len(env_ids), device=self.device) * self._trajectory.cfg.period
        reference_pos = self._trajectory.position(self._trajectory_time[env_ids])
        # Reset robot state near the reference, level and at rest
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] = reference_pos + self._terrain.env_origins[env_ids]
        default_root_state[:, :3] += torch.empty_like(reference_pos).uniform_(
            -self.cfg.init_pos_noise, self.cfg.init_pos_noise
        )
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first time
        if debug_vis:
            if not hasattr(self, "reference_visualizer"):
                self.reference_visualizer = VisualizationMarkers(
                    VisualizationMarkersCfg(
                        prim_path="/Visuals/Command/reference",
                        markers={
                            "sphere": sim_utils.SphereCfg(
                                radius=0.03,
                                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                            )
                        },
                    )
                )
                self.path_visualizer = VisualizationMarkers(
                    VisualizationMarkersCfg(
                        prim_path="/Visuals/Command/path",
                        markers={
                            "sphere": sim_utils.SphereCfg(
                                radius=0.008,
                                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.6, 1.0)),
                            )
                        },
                    )
                )
            # set their visibility to true
            self.reference_visualizer.set_visibility(True)
            self.path_visualizer.set_visibility(True)
            self._path_needs_update = True
        else:
            if hasattr(self, "reference_visualizer"):
                self.reference_visualizer.set_visibility(False)
                self.path_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # update the markers
        self.reference_visualizer.visualize(self._trajectory.position(self._trajectory_time) + self._terrain.env_origins)
        if self._path_needs_update:
            times = torch.linspace(0.0, self._trajectory.cfg.period, NUM_PATH_MARKERS + 1, device=self.device)[:-1]
            path = self._trajectory.position(times).unsqueeze(0) + self._terrain.env_origins.unsqueeze(1)
            self.path_visualizer.visualize(path.reshape(-1, 3))
            self._path_needs_update = False
