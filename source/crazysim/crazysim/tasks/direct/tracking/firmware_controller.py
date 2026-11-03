"""Crazyflie 2.x firmware position controller, batched over environments.

Port of the default PID cascade of bitcraze/crazyflie-firmware (master): ``position_controller_pid.c``,
``attitude_pid_controller.c``, ``controller_pid.c``, ``pid.c``, ``power_distribution_quadrotor.c`` and the
thrust compensation in ``motors.c``. Gains and limits are the defaults of ``platform_defaults_cf2.h``.

    position setpoint -> position PID (100 Hz) -> velocity PID (100 Hz) -> roll, pitch, thrust
    roll, pitch, yaw = 0 -> attitude PID (500 Hz) -> rate PID (500 Hz) -> motor commands -> motor thrusts

Units and signs follow the firmware: degrees, the legacy pitch angle (positive nose up, so the rate
controller sees the negated y gyro), and motor commands in [0, 65535].

Differences from the firmware:

- Inputs are the true state; the firmware uses the Kalman estimate from motion capture and the IMU.
- The gyro low-pass filter (80 Hz) runs at the 500 Hz of the rate loop instead of 1 kHz.
- The integer truncation of motor commands is not modeled.
- The motor lag is not part of the firmware. It is a first-order model of the rotors and is needed: without
  it the rate loop's derivative term is unstable at 500 Hz, because the gain was tuned on a drone with lag.

This module only depends on torch so it can be tested without Isaac Sim.
"""

from __future__ import annotations

import math
import torch

POSITION_RATE_DIVIDER = 5
"""Attitude loop steps (500 Hz) per position loop step (100 Hz)."""

UINT16_MAX = 65535.0
INT16_MAX = 32767.0
THRUST_SCALE = 1000.0
THRUST_BASE = 36000.0
THRUST_MIN = 20000.0
ROLL_PITCH_LIMIT = 20.0
"""``PID_VEL_ROLL_MAX``, ``PID_VEL_PITCH_MAX`` [deg]."""
ROLL_PITCH_LIMIT_OVERHEAD = 1.1
VEL_MAX_OVERHEAD = 1.1
DEFAULT_INTEGRATION_LIMIT = 5000.0
GYRO_LPF_CUTOFF = 80.0


class AxisPid:
    """``pidUpdate`` of ``pid.c`` for three axes of a batch. The firmware's D filter is disabled by default."""

    def __init__(self, num_envs, device, dt, kp, ki, kd, i_limit=None, output_limit=None):
        self.dt = dt
        self.kp = torch.tensor(kp, device=device)
        self.ki = torch.tensor(ki, device=device)
        self.kd = torch.tensor(kd, device=device)
        limit = (DEFAULT_INTEGRATION_LIMIT,) * 3 if i_limit is None else i_limit
        self.i_limit = torch.tensor(limit, device=device)
        self.output_limit = None if output_limit is None else torch.tensor(output_limit, device=device)
        self.integ = torch.zeros(num_envs, 3, device=device)
        self.prev_measured = torch.zeros(num_envs, 3, device=device)

    def update(self, desired: torch.Tensor, measured: torch.Tensor, wrap_last_axis: bool = False) -> torch.Tensor:
        error = desired - measured
        delta = -(measured - self.prev_measured)
        if wrap_last_axis:
            # the yaw angle: take the short way around at +-180 deg
            error = error.clone()
            delta = delta.clone()
            error[:, 2] = _wrap_180(error[:, 2])
            delta[:, 2] = _wrap_180(delta[:, 2])
        self.integ = torch.clamp(self.integ + error * self.dt, -self.i_limit, self.i_limit)
        output = self.kp * error + self.kd * delta / self.dt + self.ki * self.integ
        self.prev_measured = measured.clone()
        if self.output_limit is not None:
            output = torch.clamp(output, -self.output_limit, self.output_limit)
        return output

    def reset(self, env_ids: torch.Tensor, measured: torch.Tensor | float = 0.0):
        self.integ[env_ids] = 0.0
        self.prev_measured[env_ids] = measured


def _wrap_180(angle: torch.Tensor) -> torch.Tensor:
    return torch.where(angle > 180.0, angle - 360.0, torch.where(angle < -180.0, angle + 360.0, angle))


class GyroFilter:
    """Second-order Butterworth low-pass filter (``lpf2p`` of the firmware)."""

    def __init__(self, num_envs, device, sample_rate, cutoff):
        ohm = math.tan(math.pi * cutoff / sample_rate)
        cos_pi_4 = math.cos(math.pi / 4.0)
        c = 1.0 + 2.0 * cos_pi_4 * ohm + ohm * ohm
        self.b0 = ohm * ohm / c
        self.b1 = 2.0 * self.b0
        self.b2 = self.b0
        self.a1 = 2.0 * (ohm * ohm - 1.0) / c
        self.a2 = (1.0 - 2.0 * cos_pi_4 * ohm + ohm * ohm) / c
        self.d1 = torch.zeros(num_envs, 3, device=device)
        self.d2 = torch.zeros(num_envs, 3, device=device)

    def apply(self, sample: torch.Tensor) -> torch.Tensor:
        d0 = sample - self.d1 * self.a1 - self.d2 * self.a2
        output = d0 * self.b0 + self.d1 * self.b1 + self.d2 * self.b2
        self.d2 = self.d1
        self.d1 = d0
        return output

    def reset(self, env_ids: torch.Tensor):
        self.d1[env_ids] = 0.0
        self.d2[env_ids] = 0.0


class FirmwarePositionController:
    """One instance runs the controller of every environment. Call :meth:`step` at every physics step (500 Hz)."""

    def __init__(
        self,
        num_envs: int,
        device,
        physics_dt: float,
        motor_thrust_max: float,
        motor_thrust_min: float,
        motor_time_constant: float,
        xy_velocity_max: float,
        z_velocity_max: float,
    ):
        self.num_envs = num_envs
        self.device = device
        self.motor_thrust_max = motor_thrust_max
        self.motor_thrust_min = motor_thrust_min
        self.motor_step = 1.0 - math.exp(-physics_dt / motor_time_constant)
        self._tick = 0
        position_dt = physics_dt * POSITION_RATE_DIVIDER

        vel_limit = (
            xy_velocity_max * VEL_MAX_OVERHEAD,
            xy_velocity_max * VEL_MAX_OVERHEAD,
            max(z_velocity_max, 0.5) * VEL_MAX_OVERHEAD,
        )
        tilt_limit = ROLL_PITCH_LIMIT * ROLL_PITCH_LIMIT_OVERHEAD
        self.position_pid = AxisPid(
            num_envs, device, position_dt, (2.0, 2.0, 2.0), (0.0, 0.0, 0.5), (0.0, 0.0, 0.0), output_limit=vel_limit
        )
        self.velocity_pid = AxisPid(
            num_envs,
            device,
            position_dt,
            (25.0, 25.0, 25.0),
            (1.0, 1.0, 15.0),
            (0.0, 0.0, 0.0),
            output_limit=(tilt_limit, tilt_limit, UINT16_MAX / 2.0 / THRUST_SCALE),
        )
        self.attitude_pid = AxisPid(
            num_envs, device, physics_dt, (6.0, 6.0, 6.0), (3.0, 3.0, 1.0), (0.0, 0.0, 0.35), i_limit=(20.0, 20.0, 360.0)
        )
        self.rate_pid = AxisPid(
            num_envs, device, physics_dt, (250.0, 250.0, 120.0), (500.0, 500.0, 16.7), (2.5, 2.5, 0.0),
            i_limit=(33.3, 33.3, 166.7),
        )
        self.gyro_filter = GyroFilter(num_envs, device, 1.0 / physics_dt, GYRO_LPF_CUTOFF)

        self.desired_roll_pitch = torch.zeros(num_envs, 2, device=device)
        self.thrust_command = torch.full((num_envs,), THRUST_BASE, device=device)
        self.motor_thrust = torch.zeros(num_envs, 4, device=device)
        """Thrust of each motor (firmware order m1..m4) [N]."""

    def reset(self, env_ids: torch.Tensor, hover_motor_thrust: float, position: torch.Tensor):
        """Reset to a trimmed hover: the real drone starts tracking after a take-off and a 1 s hold."""
        self.position_pid.reset(env_ids, position)
        self.velocity_pid.reset(env_ids)
        self.attitude_pid.reset(env_ids)
        self.rate_pid.reset(env_ids)
        self.gyro_filter.reset(env_ids)
        self.desired_roll_pitch[env_ids] = 0.0
        hover_command = hover_motor_thrust / self.motor_thrust_max * UINT16_MAX
        # the z velocity integrator holds the difference between THRUST_BASE and the hover command
        self.velocity_pid.integ[env_ids, 2] = (hover_command - THRUST_BASE) / THRUST_SCALE / self.velocity_pid.ki[2]
        self.thrust_command[env_ids] = hover_command
        self.motor_thrust[env_ids] = hover_motor_thrust

    def _position_controller(self, setpoint, position, velocity, yaw_deg):
        """``positionController`` and ``velocityController``: position setpoint to attitude and thrust commands."""
        cos_yaw = torch.cos(torch.deg2rad(yaw_deg))
        sin_yaw = torch.sin(torch.deg2rad(yaw_deg))

        def to_yaw_frame(vector):
            return torch.stack(
                [
                    vector[:, 0] * cos_yaw + vector[:, 1] * sin_yaw,
                    -vector[:, 0] * sin_yaw + vector[:, 1] * cos_yaw,
                    vector[:, 2],
                ],
                dim=-1,
            )

        desired_velocity = self.position_pid.update(to_yaw_frame(setpoint), to_yaw_frame(position))
        output = self.velocity_pid.update(desired_velocity, to_yaw_frame(velocity))
        pitch = torch.clamp(-output[:, 0], -ROLL_PITCH_LIMIT, ROLL_PITCH_LIMIT)
        roll = torch.clamp(-output[:, 1], -ROLL_PITCH_LIMIT, ROLL_PITCH_LIMIT)
        self.desired_roll_pitch = torch.stack([roll, pitch], dim=-1)
        thrust = output[:, 2] * THRUST_SCALE + THRUST_BASE
        self.thrust_command = torch.clamp(thrust, min=THRUST_MIN).clamp(0.0, UINT16_MAX)

    def _motor_commands(self, attitude_deg, gyro_deg):
        """``attitudeController`` and ``powerDistributionLegacy`` with capping: motor commands of the 500 Hz loop."""
        desired = torch.cat([self.desired_roll_pitch, torch.zeros(self.num_envs, 1, device=self.device)], dim=-1)
        # legacy convention: the attitude pitch is positive nose up, the gyro y is positive nose down
        measured_attitude = attitude_deg * torch.tensor([1.0, -1.0, 1.0], device=self.device)
        desired_rate = self.attitude_pid.update(desired, measured_attitude, wrap_last_axis=True)
        filtered_gyro = self.gyro_filter.apply(gyro_deg) * torch.tensor([1.0, -1.0, 1.0], device=self.device)
        output = torch.clamp(self.rate_pid.update(desired_rate, filtered_gyro), -INT16_MAX, INT16_MAX)

        r = output[:, 0] / 2.0
        p = output[:, 1] / 2.0
        y = -output[:, 2]
        t = self.thrust_command
        motors = torch.stack([t - r + p + y, t - r - p - y, t + r - p + y, t + r + p - y], dim=-1)
        # powerDistributionCap: lower all motors equally when one exceeds the maximum, which keeps the torque
        reduction = (motors.max(dim=1, keepdim=True).values - UINT16_MAX).clamp(min=0.0)
        return (motors - reduction).clamp(0.0, UINT16_MAX)

    def step(
        self,
        setpoint: torch.Tensor,
        position: torch.Tensor,
        velocity_w: torch.Tensor,
        attitude_deg: torch.Tensor,
        gyro_deg: torch.Tensor,
    ) -> torch.Tensor:
        """Advance one 500 Hz step. Returns the thrust of each motor [N].

        Args:
            setpoint: position setpoint, shape (N, 3) [m].
            position: position in the frame of the setpoint, shape (N, 3) [m].
            velocity_w: world velocity, shape (N, 3) [m/s].
            attitude_deg: roll, pitch, yaw with the pitch positive nose down (right-handed), shape (N, 3) [deg].
            gyro_deg: body rates about x, y, z, shape (N, 3) [deg/s].
        """
        if self._tick % POSITION_RATE_DIVIDER == 0:
            self._position_controller(setpoint, position, velocity_w, attitude_deg[:, 2])
        self._tick += 1
        motors = self._motor_commands(attitude_deg, gyro_deg)
        # the compensated command is linear in thrust; below the minimum the firmware stops the motor
        target = motors / UINT16_MAX * self.motor_thrust_max
        target = torch.where(target < self.motor_thrust_min, torch.zeros_like(target), target)
        self.motor_thrust = self.motor_thrust + self.motor_step * (target - self.motor_thrust)
        return self.motor_thrust
