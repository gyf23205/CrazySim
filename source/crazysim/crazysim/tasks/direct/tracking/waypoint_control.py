"""Observation, action and reward of the trajectory-setpoint controller of ``Drone_Test_TRO``.

Batched torch versions of ``examples/trajectory_setpoints/observation.py`` and of the command constraints and
``compute_reward_components`` in ``examples/setpoint/cf_setpoint_ppo_real_common.py``. This module only depends
on torch so ``scripts/skrl/check_waypoint_contract.py`` can compare it against that repository directly.
"""

from __future__ import annotations

import math
import torch


def build_observation(
    position: torch.Tensor,
    velocity_w: torch.Tensor,
    setpoints: torch.Tensor,
    phase: torch.Tensor,
    applied_command: torch.Tensor,
    position_scale: torch.Tensor,
    velocity_scale: float,
    offset_limit: torch.Tensor,
) -> torch.Tensor:
    """Observation with ``3 N + 8`` values; ``setpoints`` has shape (N, 3) and ``phase`` is in [0, 1)."""
    relative_setpoints = (setpoints.unsqueeze(0) - position.unsqueeze(1)) / position_scale
    angle = 2.0 * math.pi * phase
    return torch.cat(
        [
            relative_setpoints.flatten(1),
            torch.sin(angle).unsqueeze(1),
            torch.cos(angle).unsqueeze(1),
            velocity_w / velocity_scale,
            (applied_command - position) / offset_limit,
        ],
        dim=-1,
    )


def action_to_waypoint(action: torch.Tensor, position: torch.Tensor, offset_limit: torch.Tensor) -> torch.Tensor:
    return position + action * offset_limit


def waypoint_to_action(target: torch.Tensor, position: torch.Tensor, offset_limit: torch.Tensor) -> torch.Tensor:
    return torch.clamp((target - position) / offset_limit, -1.0, 1.0)


def clamp_to_geofence(command: torch.Tensor, xy_limit: float, z_min: float, z_max: float) -> torch.Tensor:
    return torch.stack(
        [
            command[:, 0].clamp(-xy_limit, xy_limit),
            command[:, 1].clamp(-xy_limit, xy_limit),
            command[:, 2].clamp(z_min, z_max),
        ],
        dim=-1,
    )


def slew_limit(target: torch.Tensor, previous: torch.Tensor, rate_limit: torch.Tensor, dt: float) -> torch.Tensor:
    """``_rate_limit_command``: move the applied setpoint toward the target at a limited speed."""
    max_step = rate_limit * dt
    return previous + torch.clamp(target - previous, -max_step, max_step)


def proximity_limit(target: torch.Tensor, position: torch.Tensor, radius: torch.Tensor) -> torch.Tensor:
    """``_proximity_limit_command``: confine the applied setpoint to a box around the drone."""
    return position + torch.clamp(target - position, -radius, radius)


def reward_components(
    position: torch.Tensor,
    velocity_w: torch.Tensor,
    reference_position: torch.Tensor,
    reference_velocity: torch.Tensor,
    action: torch.Tensor,
    desired_action: torch.Tensor,
    dt: float,
    position_precision_scale: torch.Tensor,
    velocity_precision_scale: torch.Tensor,
    coarse_position_distance: float,
    velocity_gate_distance: float,
    alive_scale: float,
    coarse_position_scale: float,
    precise_position_scale: float,
    velocity_scale: float,
    waypoint_error_scale: float,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Per-step reward terms (without the crash penalty and success bonus) and tracking metrics."""
    position_error_w = position - reference_position
    velocity_error_w = velocity_w - reference_velocity
    position_error = torch.linalg.norm(position_error_w, dim=1)
    velocity_error = torch.linalg.norm(velocity_error_w, dim=1)

    coarse_quality = 1.0 - torch.clamp(position_error / coarse_position_distance, 0.0, 1.0)
    precise_quality = torch.exp(-torch.sum(torch.square(position_error_w / position_precision_scale), dim=1))
    velocity_quality = torch.exp(-torch.sum(torch.square(velocity_error_w / velocity_precision_scale), dim=1))
    velocity_gate = torch.exp(-torch.square(position_error / velocity_gate_distance))
    waypoint_action_cost = torch.mean(torch.square(action - desired_action), dim=1) / 4.0

    desired_speed = torch.linalg.norm(reference_velocity, dim=1, keepdim=True)
    tangent = reference_velocity / desired_speed.clamp(min=1e-6)
    along_error = torch.sum(position_error_w * tangent, dim=1, keepdim=True)
    cross_error = torch.linalg.norm(position_error_w - along_error * tangent, dim=1)
    cross_error = torch.where(desired_speed.squeeze(1) > 1e-6, cross_error, position_error)

    terms = {
        "alive": torch.full_like(position_error, alive_scale * dt),
        "coarse_position": coarse_position_scale * coarse_quality * dt,
        "precise_position": precise_position_scale * precise_quality * dt,
        "velocity": velocity_scale * velocity_gate * velocity_quality * dt,
        "waypoint_error": waypoint_error_scale * waypoint_action_cost * dt,
    }
    metrics = {"position_error": position_error, "cross_error": cross_error}
    return terms, metrics
