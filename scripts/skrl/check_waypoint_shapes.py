"""Compare the simulated reference shapes with the table written by ``check_waypoint_contract.py``.

Needs the Isaac Lab environment:

    python scripts/skrl/check_waypoint_shapes.py waypoint_shapes.npz
"""

import sys

import numpy as np
import torch

from isaaclab.app import AppLauncher

simulation_app = AppLauncher(headless=True).app

from crazysim.tasks.direct.tracking.waypoint_env_cfg import WaypointTrackingEnvCfg  # noqa: E402

table = np.load(sys.argv[1])
cfg = WaypointTrackingEnvCfg()
print(f"{'shape':<10}{'period sim':>12}{'period real':>12}{'position':>12}{'velocity':>12}{'setpoints':>12}", flush=True)
for name, task in cfg.tasks.items():
    trajectory = task.class_type(task)
    times = torch.tensor(table[f"{name}_times"], dtype=torch.float64)
    setpoint_times = torch.linspace(0.0, task.period, cfg.setpoint_count + 1, dtype=torch.float64)[:-1]
    position = np.abs(trajectory.position(times).numpy() - table[f"{name}_position"]).max()
    velocity = np.abs(trajectory.velocity(times).numpy() - table[f"{name}_velocity"]).max()
    setpoints = np.abs(trajectory.position(setpoint_times).numpy() - table[f"{name}_setpoints"]).max()
    print(f"{name:<10}{task.period:>12.5f}{float(table[f'{name}_period']):>12.5f}{position:>12.2e}{velocity:>12.2e}{setpoints:>12.2e}", flush=True)
simulation_app.close()
