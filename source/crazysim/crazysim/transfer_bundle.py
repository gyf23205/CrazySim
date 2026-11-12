"""File format of the learner handoff between this repository and ``Drone_Test_TRO``.

One ``transfer_bundle.pt`` holds a learner as tensors, numbers, strings, lists and mappings only, so it
loads with ``torch.load(..., weights_only=True)``. The format is version 1 of ``TRANSFER_BUNDLE.md`` in
``Drone_Test_TRO``; ``examples/setpoint/transfer_bundle.py`` there is the reference implementation. This
module only reads, checks and writes the file. :mod:`crazysim.warm_start` translates the methods of this
repository to and from its contents.

Needs only torch, not Isaac Sim.
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
import math
import torch
from collections.abc import Mapping
from pathlib import Path

VERSION = 1
ARTIFACT_TYPE = "simulation_real_continuation"
BUNDLE_FILE = "transfer_bundle.pt"
SHAPES = ("square", "circle", "triangle", "figure8", "skewed_figure8")
EQUATIONS = {
    "kae_moe": "crazysim_kae_moe_v2",
    "residual": "crazysim_residual_v2",
    "componet": "crazysim_componet_v2",
    "finetune": "crazysim_waypoint_ppo_v1",
    "fame": "crazysim_fame_585e33a_v2",
}
"""Equation identifier that ``Drone_Test_TRO`` requires for each method."""
OBSERVATION_SPECIFICATION = {
    "setpoints": 16,
    "reference_values": 48,
    "order": ["relative_reference_xyz", "phase_sin_cos", "scaled_world_velocity_xyz", "scaled_applied_offset_xyz"],
}
ENVIRONMENT_FIELDS = (
    "control_period_s", "target_height", "trajectory_scale", "trajectory_speed", "trajectory_laps",
    "square_side_length", "circle_radius", "triangle_side_length", "figure8_amp_x", "figure8_amp_y",
    "figure8_random_height", "observation_mode", "trajectory_setpoint_count", "waypoint_offset_limit",
    "observation_position_error_scale", "observation_velocity_scale", "action_clip_min", "action_clip_max",
    "command_constraint", "command_rate_limit", "command_proximity_radius", "position_cmd_xy_min",
    "position_cmd_xy_max", "position_cmd_z_min", "position_cmd_z_max",
)  # fmt: skip
"""Task settings that the importer of ``Drone_Test_TRO`` compares with its own run."""


def plain(value):
    """Copy ``value`` with tensors detached on the CPU and tuples as lists; reject anything that needs pickling."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or type(value) in (str, bool, int, float):
        return value
    raise ValueError(f"A bundle cannot hold a value of type {type(value).__name__}.")


def digest(value) -> str:
    """SHA-256 over the canonical traversal of ``TRANSFER_BUNDLE.md`` (mapping keys sorted, lists in order)."""
    sha = hashlib.sha256()

    def visit(item):
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            sha.update(f"tensor:{tensor.dtype}:{tuple(tensor.shape)}:".encode())
            sha.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, Mapping):
            sha.update(f"mapping:{len(item)}:".encode())
            for key in sorted(item, key=lambda k: (type(k).__name__, str(k))):
                visit(key)
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            sha.update(f"list:{len(item)}:".encode())
            for child in item:
                visit(child)
        elif item is None or type(item) in (str, bool, int, float):
            sha.update(f"{type(item).__name__}:".encode())
            sha.update(json.dumps(item, allow_nan=False).encode())
            sha.update(b";")
        else:
            raise ValueError(f"A bundle cannot hold a value of type {type(item).__name__}.")

    visit(value)
    return sha.hexdigest()


def check_finite(value):
    """Raise if any tensor or float in ``value`` is NaN or infinite."""
    if isinstance(value, torch.Tensor):
        if not torch.isfinite(value).all():
            raise ValueError("The bundle contains a non-finite tensor.")
    elif isinstance(value, Mapping):
        for item in value.values():
            check_finite(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            check_finite(item)
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError("The bundle contains a non-finite number.")


def dependency_records(policy_node: Mapping) -> list[dict]:
    """Frozen earlier policies and KAEs inside a policy description, in order, with their checksums."""
    records = []

    def visit(node, location="policy"):
        kind = node["kind"]
        if kind == "StackedPortableKAE":
            for index, (origin, block) in enumerate(zip(node["origins"], node["blocks"])):
                records.append({"identity": origin, "sha256": digest(block), "kind": "kae", "index": index})
        elif kind in ("ResidualPolicy", "FrozenBaseResidualCore"):
            records.append(
                {"identity": location + ".base", "sha256": digest(node["base"]), "kind": "actor", "index": 0}
            )
        elif kind in ("CompoNetPolicy", "CompoNetCore"):
            for index, prior in enumerate(node["priors"]):
                records.append(
                    {"identity": f"{location}.prior.{index}", "sha256": digest(prior), "kind": "actor", "index": index}
                )
        if "kae" in node:
            visit(node["kae"], location + ".kae")

    visit(policy_node)
    return records


def distribution_contract(method: str, policy_node: Mapping) -> dict:
    """How the action noise is applied, as ``Drone_Test_TRO`` records it."""
    if method == "fame":
        return {
            "kind": "state_dependent_raw_mean_tanh_gaussian_v1",
            "log_std_min": -5.0,
            "log_std_max": 2.0,
            "log_probability": "raw_action_with_tanh_jacobian",
        }
    return {
        "kind": "bounded_center_inverse_tanh_gaussian_v1",
        "epsilon": 1e-6,
        "min_log_std": policy_node["min_log_std"],
        "log_probability": "inverse_tanh_with_tanh_jacobian",
    }


def write_bundle(bundle: Mapping, path: str | Path) -> Path:
    """Check ``bundle``, give it its content checksum (``manifest.artifact_id``) and write it to ``path``."""
    bundle = plain(bundle)
    bundle.pop("_source", None)
    check_finite(bundle)
    manifest = bundle["manifest"]
    manifest.pop("artifact_id", None)
    manifest["artifact_id"] = digest(bundle)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(bundle, temporary)
    temporary.replace(path)
    read_bundle(path)
    return path


def read_bundle(path: str | Path) -> dict:
    """Load a bundle and check its version, checksum, method and sizes.

    The result has an extra ``_source`` entry with the file path and the SHA-256 of the file.
    """
    source = Path(path).expanduser().resolve()
    raw = source.read_bytes()
    bundle = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    try:
        manifest = bundle["manifest"]
        if manifest["version"] != VERSION or manifest["artifact_type"] != ARTIFACT_TYPE:
            raise ValueError("Unsupported bundle version or type.")
        unsealed = copy.deepcopy(bundle)
        del unsealed["manifest"]["artifact_id"]
        if manifest["artifact_id"] != digest(unsealed):
            raise ValueError("The bundle checksum does not match its contents.")
        check_finite(bundle)
        method = manifest["method"]
        if method not in EQUATIONS or manifest["equations"] != EQUATIONS[method]:
            raise ValueError("The bundle uses an unsupported method or equation version.")
        if (manifest["observation_dim"], manifest["action_dim"]) != (56, 3):
            raise ValueError("The bundle is not for 56 observations and 3 actions.")
        if manifest["shape"] not in SHAPES or manifest["phase"] not in ("simulation", "real", "kinematic"):
            raise ValueError("The bundle has an unsupported shape or phase.")
        for role in ("policy", "value"):
            if digest(bundle["architecture"][role]["state"]) != digest(bundle["agent_modules"][role]):
                raise ValueError(f"The {role} description and its saved state disagree.")
        if manifest["dependencies"] != dependency_records(bundle["architecture"]["policy"]):
            raise ValueError("The recorded frozen dependencies do not match the policy description.")
    except (KeyError, TypeError, AttributeError, IndexError) as error:
        raise ValueError(f"Invalid transfer bundle {source}: {error!r}") from error
    bundle["_source"] = {"path": str(source), "sha256": hashlib.sha256(raw).hexdigest()}
    return bundle
