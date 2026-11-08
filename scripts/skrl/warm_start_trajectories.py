"""Additive reference trajectories used only by the warm-start launcher.

The native factories and configuration classes remain unchanged. Circle and
figure-eight velocities use Drone_Test_TRO's analytic tangents; the skewed
figure-eight rotates its position and tangent +30 degrees about world Y.
"""

from __future__ import annotations

import math

import torch

from crazysim.tasks.direct.tracking.trajectories import (
    ArcLengthFigure8, ArcLengthFigure8Cfg, Circle, CircleCfg,
)


def figure8_path_length(a: float, b: float, samples: int = 4096) -> float:
    """The real runtime's midpoint integration, rather than chord length."""
    step = 2.0 * math.pi / samples
    return sum(math.hypot(a * math.cos((i + .5) * step),
                          2.0 * b * math.cos(2.0 * (i + .5) * step)) * step
               for i in range(samples))


class MatchingCircle(Circle):
    def velocity(self, t: torch.Tensor, dt: float = 1.0e-2) -> torch.Tensor:
        del dt
        angle = 2.0 * math.pi * torch.remainder(t / self.cfg.period, 1.0)
        speed = self.cfg.radius * 2.0 * math.pi / self.cfg.period
        velocity = torch.zeros(*t.shape, 3, dtype=t.dtype, device=t.device)
        velocity[..., 0] = -speed * torch.sin(angle)
        velocity[..., 1] = speed * torch.cos(angle)
        return velocity


class MatchingArcLengthFigure8(ArcLengthFigure8):
    def _angle(self, phase: torch.Tensor) -> torch.Tensor:
        # Reuse the native trajectory's exact same arc-length table/order.
        theta = self._theta.to(device=phase.device, dtype=phase.dtype)
        arc = self._arc.to(device=phase.device, dtype=phase.dtype)
        distance = phase * arc[-1]
        upper = torch.searchsorted(arc, distance.contiguous(), right=True).clamp(1, len(arc) - 1)
        lower = upper - 1
        alpha = (distance - arc[lower]) / (arc[upper] - arc[lower]).clamp(min=1e-8)
        return theta[lower] + alpha * (theta[upper] - theta[lower])

    def velocity(self, t: torch.Tensor, dt: float = 1.0e-2) -> torch.Tensor:
        del dt
        theta = self._angle(torch.remainder(t / self.cfg.period, 1.0))
        a, b = self.cfg.size
        dx, dy = a * torch.cos(theta), 2.0 * b * torch.cos(2.0 * theta)
        norm = torch.sqrt(dx.square() + dy.square()).clamp(min=1e-8)
        speed = figure8_path_length(a, b) / self.cfg.period
        velocity = torch.zeros(*t.shape, 3, dtype=t.dtype, device=t.device)
        velocity[..., 0], velocity[..., 1] = speed * dx / norm, speed * dy / norm
        return velocity


def _rotate_y(vector: torch.Tensor) -> torch.Tensor:
    cosine, sine = math.cos(math.pi / 6.0), math.sin(math.pi / 6.0)
    result = vector.clone()
    result[..., 0] = cosine * vector[..., 0] + sine * vector[..., 2]
    result[..., 2] = -sine * vector[..., 0] + cosine * vector[..., 2]
    return result


class SkewedArcLengthFigure8(MatchingArcLengthFigure8):
    """Positive-X lobe slopes downward, matching the real +30° Y rotation."""

    def position(self, t: torch.Tensor) -> torch.Tensor:
        center = torch.tensor(self.cfg.center, dtype=t.dtype, device=t.device)
        return center + _rotate_y(super().position(t) - center)

    def velocity(self, t: torch.Tensor, dt: float = 1.0e-2) -> torch.Tensor:
        return _rotate_y(super().velocity(t, dt))


def matching_circle_config(*, radius, center, period):
    return CircleCfg(class_type=MatchingCircle, radius=radius, center=center, period=period)


def matching_figure8_config(*, size, center, period, skewed=False):
    return ArcLengthFigure8Cfg(class_type=SkewedArcLengthFigure8 if skewed else MatchingArcLengthFigure8,
                              size=size, center=center, period=period)
