"""Reference trajectories for the tracking tasks.

Each trajectory is a closed planar curve, traversed once per ``period``. The curve lies in the plane
through ``center`` set by ``plane`` (horizontal ``"xy"`` or vertical ``"xz"`` / ``"yz"``).
Positions are expressed relative to the environment origin.

To add a task, subclass :class:`Trajectory`, implement :meth:`Trajectory.curve`, write a matching
config class, and add an entry to ``TrackingEnvCfg.tasks``.
"""

from __future__ import annotations

import math
import torch
from dataclasses import MISSING

from isaaclab.utils import configclass

PLANE_AXES = {"xy": [0, 1], "xz": [0, 2], "yz": [1, 2]}
"""World axes that the two in-plane coordinates of a curve map to, for each ``TrajectoryCfg.plane``."""


class Trajectory:
    """Base class: maps time to a reference position and velocity."""

    cfg: TrajectoryCfg

    def __init__(self, cfg: TrajectoryCfg):
        self.cfg = cfg

    def curve(self, phase: torch.Tensor) -> torch.Tensor:
        """In-plane offset from the center for ``phase`` in [0, 1). Returns shape ``(*phase.shape, 2)``."""
        raise NotImplementedError

    def position(self, t: torch.Tensor) -> torch.Tensor:
        """Reference position at time ``t`` [s]. Returns shape ``(*t.shape, 3)``."""
        phase = torch.remainder(t / self.cfg.period, 1.0)
        center = torch.tensor(self.cfg.center, device=t.device, dtype=t.dtype)
        offset = torch.zeros(*t.shape, 3, device=t.device, dtype=t.dtype)
        offset[..., PLANE_AXES[self.cfg.plane]] = self.curve(phase)
        return center + offset

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


class Star(Trajectory):
    """Star polygon traversed at constant speed, starting at the outer tip on the first in-plane axis.

    The vertices alternate between the outer radius (tips) and the inner radius (notches).
    """

    cfg: StarCfg

    def __init__(self, cfg: StarCfg):
        super().__init__(cfg)
        n = cfg.num_tips
        angles = math.pi * torch.arange(2 * n + 1) / n
        radii = torch.tensor([cfg.outer_radius, cfg.inner_radius]).repeat(n + 1)[: 2 * n + 1]
        # closed polygon: the last vertex repeats the first
        self._vertices = radii.unsqueeze(-1) * torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)

    def curve(self, phase: torch.Tensor) -> torch.Tensor:
        vertices = self._vertices.to(device=phase.device, dtype=phase.dtype)
        edges = vertices[1:] - vertices[:-1]
        lengths = torch.linalg.norm(edges, dim=-1)
        cumulative = torch.cat([lengths.new_zeros(1), torch.cumsum(lengths, dim=0)])
        d = phase * cumulative[-1]  # arc length travelled along the perimeter
        index = (torch.searchsorted(cumulative, d.contiguous(), right=True) - 1).clamp(0, len(edges) - 1)
        fraction = ((d - cumulative[index]) / lengths[index]).unsqueeze(-1)
        return vertices[index] + fraction * edges[index]


@configclass
class TrajectoryCfg:
    class_type: type[Trajectory] = MISSING
    center: tuple[float, float, float] = (0.0, 0.0, 1.0)
    """Center of the curve relative to the environment origin [m]. Its z sets the flight height."""
    plane: str = "xy"
    """Plane of the curve: ``"xy"`` (horizontal), ``"xz"`` or ``"yz"`` (vertical)."""
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


@configclass
class StarCfg(TrajectoryCfg):
    class_type: type[Trajectory] = Star
    num_tips: int = 5
    outer_radius: float = 0.5
    """Distance from the center to each tip [m]."""
    inner_radius: float = 0.2
    """Distance from the center to each notch between two tips [m]."""
