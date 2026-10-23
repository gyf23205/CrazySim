"""Reference trajectories for the tracking tasks.

Each trajectory is a closed planar curve in a horizontal plane, traversed once per ``period``.
Positions are expressed relative to the environment origin.

To add a task, subclass :class:`Trajectory`, implement :meth:`Trajectory.curve`, write a matching
config class, and add an entry to ``TrackingEnvCfg.tasks``.
"""

from __future__ import annotations

import math
import torch
from dataclasses import MISSING

from isaaclab.utils import configclass


class Trajectory:
    """Base class: maps time to a reference position and velocity."""

    cfg: TrajectoryCfg

    def __init__(self, cfg: TrajectoryCfg):
        self.cfg = cfg

    def curve(self, phase: torch.Tensor) -> torch.Tensor:
        """Planar offset from the center for ``phase`` in [0, 1). Returns shape ``(*phase.shape, 2)``."""
        raise NotImplementedError

    def position(self, t: torch.Tensor) -> torch.Tensor:
        """Reference position at time ``t`` [s]. Returns shape ``(*t.shape, 3)``."""
        phase = torch.remainder(t / self.cfg.period, 1.0)
        center = torch.tensor(self.cfg.center, device=t.device, dtype=t.dtype)
        xy = self.curve(phase) + center[:2]
        z = torch.full_like(t, self.cfg.center[2]).unsqueeze(-1)
        return torch.cat([xy, z], dim=-1)

    def velocity(self, t: torch.Tensor, dt: float = 1.0e-2) -> torch.Tensor:
        """Reference velocity at time ``t`` [s], from a central difference."""
        return (self.position(t + dt) - self.position(t - dt)) / (2.0 * dt)


class Circle(Trajectory):
    cfg: CircleCfg

    def curve(self, phase: torch.Tensor) -> torch.Tensor:
        angle = 2.0 * math.pi * phase
        return self.cfg.radius * torch.stack([torch.cos(angle), torch.sin(angle)], dim=-1)


class Rectangle(Trajectory):
    """Counter-clockwise traversal of the perimeter at constant speed, starting at the (+x, -y) corner."""

    cfg: RectangleCfg

    def curve(self, phase: torch.Tensor) -> torch.Tensor:
        w, h = self.cfg.size
        d = phase * 2.0 * (w + h)  # arc length travelled along the perimeter
        x = torch.where(
            d < h, w / 2, torch.where(d < h + w, w / 2 - (d - h), torch.where(d < 2 * h + w, -w / 2, d - 2 * h - 1.5 * w))
        )
        y = torch.where(
            d < h, d - h / 2, torch.where(d < h + w, h / 2, torch.where(d < 2 * h + w, 1.5 * h + w - d, -h / 2))
        )
        return torch.stack([x, y], dim=-1)


class Figure8(Trajectory):
    """Lemniscate of Gerono: x = a sin(2 pi s), y = b sin(4 pi s)."""

    cfg: Figure8Cfg

    def curve(self, phase: torch.Tensor) -> torch.Tensor:
        angle = 2.0 * math.pi * phase
        a, b = self.cfg.size
        return torch.stack([a * torch.sin(angle), b * torch.sin(2.0 * angle)], dim=-1)


@configclass
class TrajectoryCfg:
    class_type: type[Trajectory] = MISSING
    center: tuple[float, float, float] = (0.0, 0.0, 1.0)
    """Center of the curve relative to the environment origin [m]. Its z sets the flight height."""
    period: float = 6.0
    """Time for one full traversal [s]."""


@configclass
class CircleCfg(TrajectoryCfg):
    class_type: type[Trajectory] = Circle
    radius: float = 0.5


@configclass
class RectangleCfg(TrajectoryCfg):
    class_type: type[Trajectory] = Rectangle
    size: tuple[float, float] = (1.0, 0.6)
    """Side lengths along x and y [m]."""


@configclass
class Figure8Cfg(TrajectoryCfg):
    class_type: type[Trajectory] = Figure8
    size: tuple[float, float] = (0.6, 0.3)
    """Half-extents along x and y [m]."""
