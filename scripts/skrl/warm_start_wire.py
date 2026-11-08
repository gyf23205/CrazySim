"""Tensor-only v1 handoff wire primitives; no simulator or real-runtime imports.

This implements the published Drone_Test_TRO TRANSFER_BUNDLE.md serialization
contract, not its algorithms. Native architecture conversion lives in
warm_start_io.py. The ordinary simulator launch paths do not import this file.
"""
from __future__ import annotations

from collections.abc import Mapping
import copy
import hashlib
import io
import json
import math
from pathlib import Path

import torch

SHAPES = ("square", "circle", "triangle", "figure8", "skewed_figure8")
EQUATIONS = {"finetune": "crazysim_waypoint_ppo_v1", "residual": "crazysim_residual_v2",
             "componet": "crazysim_componet_v2", "kae_moe": "crazysim_kae_moe_v2",
             "fame": "crazysim_fame_585e33a_v2"}
OBSERVATION_CONTRACT = "trajectory_setpoints_relative_waypoint_v1:setpoints=16:offset_limit=0.5,0.5,0.25"
OBSERVATION_SPECIFICATION = {"setpoints": 16, "reference_values": 48,
    "order": ["relative_reference_xyz", "phase_sin_cos", "scaled_world_velocity_xyz", "scaled_applied_offset_xyz"]}


def plain(value):
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
    raise ValueError(f"Nonportable state: {type(value).__name__}.")


def digest(value):
    hasher = hashlib.sha256()
    def visit(item):
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            hasher.update(f"tensor:{tensor.dtype}:{tuple(tensor.shape)}:".encode())
            hasher.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, Mapping):
            hasher.update(f"mapping:{len(item)}:".encode())
            for key in sorted(item, key=lambda k: (type(k).__name__, str(k))):
                visit(key)
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            hasher.update(f"list:{len(item)}:".encode())
            for child in item:
                visit(child)
        elif item is None or type(item) in (str, bool, int, float):
            hasher.update(f"{type(item).__name__}:".encode())
            hasher.update(json.dumps(item, allow_nan=False).encode())
            hasher.update(b";")
        else:
            raise ValueError("Unsupported wire value.")
    visit(value)
    return hasher.hexdigest()


def finite(value):
    if isinstance(value, torch.Tensor):
        if not torch.isfinite(value).all():
            raise ValueError("Transfer state contains nonfinite tensors.")
    elif isinstance(value, Mapping):
        for item in value.values():
            finite(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            finite(item)
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Transfer state contains nonfinite scalars.")


def dependencies(node):
    kind = node["kind"]
    if kind == "StackedPortableKAE":
        return [{"identity": origin, "sha256": digest(block), "kind": "kae", "index": i}
                for i, (origin, block) in enumerate(zip(node["origins"], node["blocks"]))]
    if kind in ("ExpandingKaeMoEPolicy", "ExpandingKaeMoECore"):
        return dependencies(node["kae"])
    if kind in ("ResidualPolicy", "FrozenBaseResidualCore"):
        return [{"identity": "policy.base", "sha256": digest(node["base"]), "kind": "actor", "index": 0}]
    if kind in ("CompoNetPolicy", "CompoNetCore"):
        return [{"identity": f"policy.prior.{i}", "sha256": digest(actor), "kind": "actor", "index": i}
                for i, actor in enumerate(node["priors"])]
    return []


def distribution(method):
    if method == "fame":
        return {"kind": "state_dependent_raw_mean_tanh_gaussian_v1", "log_std_min": -5.0,
                "log_std_max": 2.0, "log_probability": "raw_action_with_tanh_jacobian"}
    return {"kind": "bounded_center_inverse_tanh_gaussian_v1", "epsilon": 1e-6,
            "min_log_std": -3.0, "log_probability": "inverse_tanh_with_tanh_jacobian"}


def validate_descriptor(node):
    """Reject contradictory repeated priors, aliases or equation declarations."""
    kind, state = node["kind"], node["state"]
    child_entries = []
    if kind in ("ResidualPolicy", "FrozenBaseResidualCore"):
        child_entries.append(("residual_core.base." if kind == "ResidualPolicy" else "base.", node["base"]))
    elif kind in ("CompoNetPolicy", "CompoNetCore"):
        prefix = "componet_core.previous_units." if kind == "CompoNetPolicy" else "previous_units."
        child_entries.extend((f"{prefix}{i}.", child) for i, child in enumerate(node["priors"]))
    elif kind in ("ExpandingKaeMoEPolicy", "ExpandingKaeMoECore"):
        child_entries.append(("moe_core.kae." if kind == "ExpandingKaeMoEPolicy" else "kae.", node["kae"]))
    elif kind == "StackedPortableKAE":
        if len(node["blocks"]) != len(node["origins"]) or len(node["origins"]) != len(set(node["origins"])):
            raise ValueError("KAE bank identities must be distinct, ordered and complete.")
        child_entries.extend((f"blocks.{i}.", block) for i, block in enumerate(node["blocks"]))
    elif kind == "PortableKoopmanAutoencoder":
        child_entries.append(("encoder.normalizer.", node["normalizer"]))
    elif kind == "DeployablePolicy":
        child_entries.extend((("normalizer.", node["normalizer"]), ("actor.", node["actor"])))
    elif kind == "Sequential":
        child_entries.extend((f"{i}.", child) for i, child in enumerate(node["children"]))
    elif kind == "SquashedMean":
        child_entries.append(("actor.", node["actor"]))
    for prefix, child in child_entries:
        embedded = {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}
        if digest(embedded) != digest(child["state"]):
            raise ValueError("Embedded dependency state differs from its architecture descriptor.")
        validate_descriptor(child)
    prefix = {"ResidualPolicy": "residual_core.", "CompoNetPolicy": "componet_core.",
              "ExpandingKaeMoEPolicy": "moe_core."}.get(kind)
    if prefix is not None:
        core = {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}
        alias = {key[len("policy_layer."):]: value for key, value in state.items() if key.startswith("policy_layer.")}
        if digest(core) != digest(alias):
            raise ValueError("Portable policy alias tensors disagree.")
    method = {"ResidualPolicy": "residual", "FrozenBaseResidualCore": "residual",
              "CompoNetPolicy": "componet", "CompoNetCore": "componet", "ExpandingKaeMoEPolicy": "kae_moe",
              "ExpandingKaeMoECore": "kae_moe", "CrazySimWaypointPolicy": "finetune", "FameActor": "fame"}.get(kind)
    if method is not None and node.get("equations") != EQUATIONS[method]:
        raise ValueError("Unsupported nested actor equation version.")


def read_bundle(path):
    source = Path(path).expanduser().resolve()
    raw = source.read_bytes()
    bundle = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    try:
        manifest = bundle["manifest"]
        if manifest["version"] != 1 or manifest["artifact_type"] != "simulation_real_continuation":
            raise ValueError("Unsupported portable artifact version/type.")
        checked = copy.deepcopy(bundle)
        identity = checked["manifest"].pop("artifact_id")
        if digest(checked) != identity:
            raise ValueError("Portable artifact checksum mismatch.")
        finite(bundle)
        method = manifest["method"]
        if manifest["equations"] != EQUATIONS.get(method):
            raise ValueError("Unsupported method equation version.")
        if (manifest["observation_dim"], manifest["action_dim"]) != (56, 3):
            raise ValueError("Portable artifacts require 56 observations and 3 relative actions.")
        if manifest["observation_contract"] != OBSERVATION_CONTRACT:
            raise ValueError("Portable observation/action contract mismatch.")
        if manifest["shape"] not in SHAPES or manifest["phase"] not in ("simulation", "real", "kinematic"):
            raise ValueError("Unsupported artifact shape/phase.")
        environment = manifest["environment"]
        if (environment.get("shape") != manifest["shape"]
                or environment.get("reference_geometry_version") != "fixed_center_five_shapes_v1"
                or environment.get("shape_rotation_y_degrees") != (30.0 if manifest["shape"] == "skewed_figure8" else 0.0)
                or environment.get("command_semantics") != "world_position_setpoint_relative_waypoint_v1"):
            raise ValueError("Portable environment geometry/command contract mismatch.")
        if manifest["action_contract"] != {"bounds": [-1., 1.], "decoder": "position_plus_scaled_action",
            "offset_limit_m": environment["waypoint_offset_limit"], "yaw_radians": 0.0}:
            raise ValueError("Portable relative-waypoint action contract mismatch.")
        if manifest["observation_specification"] != OBSERVATION_SPECIFICATION or manifest["distribution"] != distribution(method):
            raise ValueError("Portable observation/distribution specification mismatch.")
        if manifest["dependencies"] != dependencies(bundle["architecture"]["policy"]):
            raise ValueError("Ordered dependency checksums mismatch.")
        for role in ("policy", "value"):
            if digest(bundle["architecture"][role]["state"]) != digest(bundle["agent_modules"][role]):
                raise ValueError("Architecture and current learner states disagree.")
            validate_descriptor(bundle["architecture"][role])
        counters = manifest["phase_counters"]
        if manifest["counters"] != counters or any(type(counters[k]) is not int or counters[k] < 0
            for k in ("accepted_transitions", "complete_updates")):
            raise ValueError("Invalid phase counters.")
        if manifest["checkpoint_selection"]["actual_update"] != counters["complete_updates"]:
            raise ValueError("Selected checkpoint update does not match learner counters.")
    except (KeyError, TypeError, RuntimeError) as error:
        raise ValueError(f"Invalid portable artifact: {error}") from error
    bundle["_source"] = {"path": str(source), "sha256": hashlib.sha256(raw).hexdigest()}
    return bundle


def write_bundle(bundle, path):
    data = plain(bundle)
    data.pop("_source", None)
    data["manifest"].pop("artifact_id", None)
    data["manifest"]["artifact_id"] = digest(data)
    finite(data)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    try:
        torch.save(data, temporary)
        read_bundle(temporary)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return target
