"""Focused native-method adapters for simulation-assisted real adaptation.

Only this dedicated launch path imports this module. Policy equations and PPO/
FAME updates come from the existing crazysim.methods implementations; this file
translates architecture/state names and portable checkpoint bookkeeping.
"""
from __future__ import annotations

from collections.abc import Mapping
import copy
import hashlib
import importlib
import math
from pathlib import Path
import pickle
import types

import torch
from torch import nn

from warm_start_wire import (EQUATIONS, OBSERVATION_CONTRACT, OBSERVATION_SPECIFICATION,
                             SHAPES, dependencies, digest, distribution, finite, plain,
                             read_bundle, write_bundle)


def _native(module):
    return importlib.import_module("crazysim.methods." + module)


def _prefix(state, prefix):
    return {prefix + key: value for key, value in state.items()}


def _strip(state, prefix):
    return {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}


def _layout(module):
    if isinstance(module, nn.Sequential):
        return [_layout(item) for item in module]
    if isinstance(module, nn.Linear):
        return {"kind": "Linear", "in_features": module.in_features, "out_features": module.out_features,
                "bias": module.bias is not None}
    if isinstance(module, nn.ELU):
        return {"kind": "ELU", "alpha": float(module.alpha)}
    if isinstance(module, (nn.Identity, nn.Tanh, nn.ReLU)):
        return {"kind": type(module).__name__}
    raise ValueError(f"Unsupported network layer {type(module).__name__}.")


def _node(kind, state=None, *, networks=None, **fields):
    return {"kind": kind, "state": plain(state or {}), "networks": networks or {}, **plain(fields)}


def _sequence(node):
    kind = node["kind"]
    if kind == "Sequential":
        module = nn.Sequential(*(_sequence(child) for child in node["children"]))
    elif kind == "Linear":
        module = nn.Linear(node["in_features"], node["out_features"], bias=node["bias"])
    elif kind == "ELU":
        module = nn.ELU(alpha=node["alpha"])
    elif kind in ("Identity", "Tanh", "ReLU"):
        module = {"Identity": nn.Identity, "Tanh": nn.Tanh, "ReLU": nn.ReLU}[kind]()
    else:
        raise ValueError(f"Unsupported elementary actor node: {kind}.")
    module.load_state_dict(node["state"], strict=True)
    return module


def _describe_sequence(module):
    name = type(module).__name__
    if isinstance(module, nn.Sequential):
        return _node("Sequential", module.state_dict(), children=[_describe_sequence(child) for child in module])
    if isinstance(module, nn.Linear):
        return _node("Linear", module.state_dict(), in_features=module.in_features,
                     out_features=module.out_features, bias=module.bias is not None)
    if isinstance(module, nn.ELU):
        return _node("ELU", alpha=module.alpha)
    if isinstance(module, (nn.Identity, nn.ReLU, nn.Tanh)):
        return _node(name)
    raise ValueError(f"Unsupported simple actor {name}.")


class FrozenInputNormalizer(nn.Module):
    """Frozen inference adapter retaining the published scaler's exact formula."""
    def __init__(self, state, epsilon=1e-8, clip_threshold=5.0):
        super().__init__()
        self.register_buffer("running_mean", state["running_mean"].detach().float().clone())
        self.register_buffer("running_variance", state["running_variance"].detach().float().clone())
        if "current_count" in state:
            self.register_buffer("current_count", state["current_count"].detach().clone())
        self.epsilon, self.clip_threshold = float(epsilon), float(clip_threshold)
        if (self.running_mean.ndim != 1 or self.running_mean.shape != self.running_variance.shape
                or not torch.isfinite(self.running_mean).all() or not torch.isfinite(self.running_variance).all()
                or (self.running_variance < 0).any() or self.epsilon <= 0 or self.clip_threshold <= 0):
            raise ValueError("Invalid frozen normalizer.")

    def forward(self, states):
        return ((states.float() - self.running_mean) /
                (self.running_variance.sqrt() + self.epsilon)).clamp(-self.clip_threshold, self.clip_threshold)


class _PaddedEncoder(nn.Module):
    def __init__(self, network, normalizer):
        super().__init__()
        self.network, self.normalizer = network, normalizer

    def forward(self, states):
        return self.network(torch.cat((self.normalizer(states[..., :56]), states[..., 56:]), dim=-1))


def _cache(module, descriptor, parameter_map=None):
    module._portable_descriptor = plain(descriptor)
    module._portable_parameter_map = parameter_map or {name: name for name, _ in module.named_parameters()}
    return module


def _normalizer_node(module):
    return _node("RunningStandardScalerModule", module.state_dict(), epsilon=module.epsilon,
                 clip_threshold=module.clip_threshold)


def _normalizer_from_node(node):
    if node["kind"] == "Identity":
        return nn.Identity()
    if node["kind"] not in ("RunningStandardScalerModule", "SingleDroneRunningStandardScaler"):
        raise ValueError("Unsupported frozen normalizer architecture.")
    return FrozenInputNormalizer(node["state"], node["epsilon"], node["clip_threshold"])


class ImportedFameActor(nn.Module):
    """Picklable input adapter delegating every distribution to the native actor."""
    def __init__(self, obs_dim, action_dim, hidden_dims, configuration=None):
        super().__init__()
        configuration = configuration or {}
        self.trunk = _native("fame").SquashedActor(obs_dim, action_dim, list(hidden_dims)).trunk
        state = configuration.get("normalization_state")
        self.input_normalizer = (nn.Identity() if state is None else FrozenInputNormalizer(
            state, configuration.get("normalizer_epsilon", 1e-8),
            configuration.get("normalizer_clip_threshold", 5.0)))
        self._portable_fame_configuration = {"hidden_dims": list(hidden_dims), **plain(configuration)}

    def dist_params(self, states):
        return _native("fame").SquashedActor.dist_params(self, self.input_normalizer(states))

    def log_prob_raw(self, states, raw_actions):
        return _native("fame").SquashedActor.log_prob_raw(self, states, raw_actions)

    def forward(self, states):
        return _native("fame").SquashedActor.forward(self, states)


def normalized_fame_actor(obs_dim, action_dim, hidden_dims, configuration=None):
    return ImportedFameActor(obs_dim, action_dim, hidden_dims, configuration)


def fame_actor_factory(bundle):
    configuration = bundle["architecture"]["policy"]["configuration"]
    return lambda obs_dim, action_dim, hidden_dims: normalized_fame_actor(
        obs_dim, action_dim, hidden_dims, configuration)


def decode_actor(node, device="cpu"):
    """Reconstruct a deterministic controller using native methods and layers."""
    kind, state = node["kind"], node["state"]
    parameter_map = {}
    if kind in ("Sequential", "Linear", "Identity", "Tanh", "ELU", "ReLU"):
        result = _sequence(node)
    elif kind in ("RunningStandardScalerModule", "SingleDroneRunningStandardScaler"):
        result = _normalizer_from_node(node)
    elif kind == "DeployablePolicy":
        result = _native("base").DeterministicPolicy(decode_actor(node["actor"], device),
                                                     decode_actor(node["normalizer"], device))
        parameter_map = {"actor." + k: "actor." + v for k, v in result.actor._portable_parameter_map.items()}
    elif kind in ("BoundedDeterministicActor", "DeterministicActor", "CrazySimWaypointPolicy"):
        if kind == "CrazySimWaypointPolicy":
            hidden = [64, 64]
        else:
            hidden = node["hidden_dims"]
        if hidden != [64, 64]:
            raise ValueError("Waypoint actor architecture must be 64/64 ELU.")
        result = _native("base").make_mlp(56, hidden, 3)
        bounded = kind != "DeterministicActor"
        if bounded:
            result.append(nn.Tanh())
        mapped = {}
        for native_prefix, canonical_prefix in (("0.", "net_container.0."), ("2.", "net_container.2."),
                                                 ("4.", "policy_layer.0." if kind == "CrazySimWaypointPolicy" else "policy_layer.")):
            for suffix in ("weight", "bias"):
                mapped[native_prefix + suffix] = state[canonical_prefix + suffix]
                parameter_map[native_prefix + suffix] = canonical_prefix + suffix
        result.load_state_dict(mapped)
    elif kind in ("ResidualPolicy", "FrozenBaseResidualCore"):
        core = _strip(state, "residual_core.") if kind == "ResidualPolicy" else state
        base = decode_actor(node["base"], device)
        residual = _native("base").make_mlp(56, [256, 128, 64], 3)
        residual.load_state_dict(_strip(core, "residual_network."))
        result = _native("residual").ComposedActor(base, residual)
        prefix = "residual_core." if kind == "ResidualPolicy" else ""
        parameter_map.update({"base." + key: prefix + "base." + value for key, value in base._portable_parameter_map.items()})
        parameter_map.update({"residual." + key: prefix + "residual_network." + key for key, _ in residual.named_parameters()})
        result._portable_chain_depth = node["chain_depth"]
    elif kind in ("CompoNetPolicy", "CompoNetCore"):
        priors = [decode_actor(prior, device) for prior in node["priors"]]
        unit = _native("componet").CompoNet(
            [_IndependentPrior(actor) for actor in priors], 56, node["hidden_dim"], 3,
            _native("base").make_mlp(56 + node["hidden_dim"], [256, 128, 64], 3), device)
        core = _strip(state, "componet_core.") if kind == "CompoNetPolicy" else state
        own = {key: value for key, value in core.items() if not key.startswith("previous_units.")}
        if unit.pe0 is None:
            own.pop("pe0", None)
        incompatible = unit.load_state_dict(own, strict=False)
        if incompatible.unexpected_keys or any(not key.startswith("previous_units.") for key in incompatible.missing_keys):
            raise ValueError("CompoNet current attention/internal-policy state is incomplete.")
        result = _native("componet").ActionOnly(unit)
        prefix = "componet_core." if kind == "CompoNetPolicy" else ""
        parameter_map = {"unit." + key: prefix + key for key, _ in unit.named_parameters()
                         if not key.startswith("previous_units.")}
        for i, actor in enumerate(priors):
            parameter_map.update({f"unit.previous_units.{i}.controller." + k: prefix + f"previous_units.{i}." + v
                                  for k, v in actor._portable_parameter_map.items()})
    elif kind in ("ExpandingKaeMoEPolicy", "ExpandingKaeMoECore"):
        bank = node["kae"]
        experts = [_expert_from_node(block, origin, device) for block, origin in zip(bank["blocks"], bank["origins"])]
        result = _native("kae_moe").MoEActor(experts, 56, 3, 57, _moe_cfg())
        core = _strip(state, "moe_core.") if kind == "ExpandingKaeMoEPolicy" else state
        head = _strip(core, "head.")
        incompatible = result.load_state_dict(head, strict=False)
        if incompatible.unexpected_keys or any(not key.startswith("kaes.") for key in incompatible.missing_keys):
            raise ValueError("MoE current router/gate/mode/MLP state is incomplete.")
        prefix = "moe_core." if kind == "ExpandingKaeMoEPolicy" else ""
        parameter_map = {key: prefix + "head." + key for key, _ in result.named_parameters() if not key.startswith("kaes.")}
        for i, expert in enumerate(experts):
            parameter_map.update({f"kaes.{i}." + key: prefix + f"kae.blocks.{i}." + value
                                  for key, value in expert._portable_parameter_map.items()})
    elif kind == "FameActor":
        result = normalized_fame_actor(56, 3, node["configuration"]["hidden_dims"], node["configuration"])
        result.load_state_dict(state)
        parameter_map = {name: name for name, _ in result.named_parameters()}
    elif kind == "SquashedMean":
        result = _native("fame").SquashedMean(decode_actor(node["actor"], device))
        parameter_map = {"actor." + k: "actor." + v for k, v in result.actor._portable_parameter_map.items()}
    elif kind in ("BoundedValue", "FameValue"):
        result = _native("base").make_mlp(56, node["hidden_dims"], 1)
        result.load_state_dict(_strip(state, "net."))
        parameter_map = {name: "net." + name for name, _ in result.named_parameters()}
    elif kind == "PortableKoopmanAutoencoder":
        result = _kae_from_node(node, device)
        parameter_map = result._portable_parameter_map
    else:
        raise ValueError(f"Unsupported portable controller kind {kind}.")
    return _cache(result.to(device), node, parameter_map or None)


class _IndependentPrior(nn.Module):
    """Bridge full real controllers to native CompoNet's phi-chain protocol."""
    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.is_prev = False
        self.requires_grad_(False)
    def forward(self, value):
        phi, states = value if isinstance(value, tuple) else (None, value)
        action = self.controller(states)
        phi = action[:, None, :] if phi is None else torch.cat((phi, action[:, None, :]), dim=1)
        return (phi, states) if self.is_prev else (action, phi)


def _moe_cfg():
    return {"residual_hidden_dims": [256, 128, 64], "weight_hidden_dims": [32],
            "gating_hidden_dims": [32], "router_hidden_dims": [128]}


def _kae_from_node(node, device):
    state = node["state"]
    if (node["padded_dim"], node["hidden_size"]) != (57, 64):
        raise ValueError("Portable KAEs must use 57 inputs and hidden size 64.")
    kae = _native("kae").KoopmanAutoencoder(57, 64, node["mode_count"], device)
    encoder = kae.encoder
    encoder.load_state_dict(_strip(state, "encoder.encoder."))
    kae.encoder = _PaddedEncoder(encoder, decode_actor(node["normalizer"], device))
    kae.decoder.load_state_dict({"weight": state["decoder.linear.weight"]})
    kae.K = state["K"].to(device)
    mapping = {"encoder.network." + key: "encoder.encoder." + key for key, _ in encoder.named_parameters()}
    mapping["decoder.weight"] = "decoder.linear.weight"
    return _cache(kae.to(device), node, mapping).requires_grad_(False).eval()


def _expert_from_node(node, origin, device):
    kae = _kae_from_node(node, device)
    expert = _native("kae").KAEExperts(kae, 3, p=1).to(device)
    # Preserve imported eigenmode identities instead of re-eigendecomposing a
    # cumulative bank. The source forward equation uses these same buffers.
    state = node["state"]
    expert.eigvals_p = state["_captured_eigenvalues"].to(device=device, dtype=torch.complex64)
    expert.eigvec_left = state["_captured_eigenvectors_inverse"].to(device=device, dtype=torch.complex64)
    expert.v = (state["decoder.linear.weight"][:3].to(torch.complex128)
                @ state["_captured_eigenvectors"]).to(device=device, dtype=torch.complex64)
    expert._portable_kae_descriptor, expert._portable_origin = plain(node), str(origin)
    expert._portable_parameter_map = {"encoder.network." + key: "encoder.encoder." + key
                                      for key, _ in kae.encoder.network.named_parameters()}
    return expert.requires_grad_(False).eval()


def completed_deterministic_controller(bundle, device="cpu"):
    node = bundle["architecture"]["policy"]
    method = bundle["manifest"]["method"]
    if method in ("residual", "componet", "kae_moe"):
        kind, prefix = {"residual": ("FrozenBaseResidualCore", "residual_core."),
                        "componet": ("CompoNetCore", "componet_core."),
                        "kae_moe": ("ExpandingKaeMoECore", "moe_core.")}[method]
        fields = {key: value for key, value in node.items()
                  if key not in ("kind", "state", "min_log_std", "policy_initialization")}
        node = {"kind": kind, "state": _strip(node["state"], prefix), **fields}
    controller = decode_actor(node, device)
    if method == "fame":
        controller = _native("fame").SquashedMean(controller)
    elif method == "finetune":
        pre = bundle["agent_modules"].get("state_preprocessor")
        if pre is not None:
            descriptor = bundle["agent_configuration"]["preprocessors"]["state_preprocessor"]
            normalizer = FrozenInputNormalizer(pre, descriptor["epsilon"], descriptor["clip_threshold"]).to(device)
            actor_node = _waypoint_actor_node(node["state"])
            wrapper_node = _node("DeployablePolicy", {**_prefix(normalizer.state_dict(), "normalizer."),
                                                     **_prefix(actor_node["state"], "actor.")},
                                 normalizer=_normalizer_node(normalizer), actor=actor_node)
            controller = decode_actor(wrapper_node, device)
    return controller.requires_grad_(False).eval()


build_predecessor_policy = completed_deterministic_controller


def build_kae_experts(bundle, device="cpu", include_append=True):
    bank = bundle["architecture"]["policy"]["kae"]
    nodes, origins = list(bank["blocks"]), list(bank["origins"])
    if include_append:
        appended = bundle["manifest"]["next_stage"]["appended_kae"]
        nodes.append(appended)
        record = bundle["manifest"]["next_stage"]["appended_kae_record"]
        origins.append("completed-real:" + record["sha256"])
    if len(origins) != len(set(origins)):
        raise ValueError("Ordered KAE bank contains duplicate identities.")
    return [_expert_from_node(node, origin, device) for node, origin in zip(nodes, origins)]


def load_real_parent(path, method, target_shape, allow_kinematic=False):
    bundle = read_bundle(path)
    manifest = bundle["manifest"]
    if manifest["method"] != method:
        raise ValueError("Parent artifact belongs to another method.")
    if target_shape not in SHAPES or SHAPES.index(target_shape) == 0 or manifest["shape"] != SHAPES[SHAPES.index(target_shape) - 1]:
        raise ValueError("Parent artifact is not the immediately preceding completed shape.")
    if manifest["phase"] != "real" and not (allow_kinematic and manifest["phase"] == "kinematic"):
        raise ValueError("Warm-up requires a completed real parent; software-only parents require --allow-kinematic.")
    next_stage = manifest.get("next_stage")
    strategy = {"finetune": "continue_actor_and_compatible_algorithm_state",
                "residual": "freeze_completed_current_controller_as_base",
                "componet": "append_completed_current_controller_to_prior_stack",
                "kae_moe": "append_full_real_policy_kae", "fame": "retain_once_consolidated_meta_state"}[method]
    if (not isinstance(next_stage, Mapping) or next_stage.get("shape") != target_shape
            or next_stage.get("phase") != "simulation" or next_stage.get("strategy") != strategy
            or not isinstance(next_stage.get("completed_real_stage"), Mapping)):
        raise ValueError("Parent must have verified completed-stage finalization for the next shape.")
    if next_stage.get("software_validation_only") and not allow_kinematic:
        raise ValueError("Software-only finalization is not a physical parent.")
    if manifest.get("software_validation_only") and not allow_kinematic:
        raise ValueError("Software-only parent lineage requires --allow-kinematic.")
    counters, budget = manifest["phase_counters"], manifest["budget"]
    rollout = bundle["agent_configuration"]["settings"].get("rollouts")
    if budget.get("completed_laps") is not None:
        # Lap budget: one PPO update per finished lap, paused laps discarded.
        if (type(budget["completed_laps"]) is not int or budget["completed_laps"] <= 0
                or counters.get("completed_laps") != budget["completed_laps"]
                or type(counters.get("complete_updates")) is not int
                or not 0 < counters["complete_updates"] <= budget["completed_laps"]):
            raise ValueError("Completed real parent has not reached its exact lap budget.")
    elif (type(budget.get("accepted_transitions")) is not int or budget["accepted_transitions"] <= 0
            or counters["accepted_transitions"] != budget["accepted_transitions"]
            or type(rollout) is not int or rollout <= 0 or counters["accepted_transitions"] % rollout
            or counters["complete_updates"] != counters["accepted_transitions"] // rollout):
        raise ValueError("Completed real parent has not reached its exact complete-update interaction budget.")
    identity = next_stage["completed_real_stage"]
    for key in ("artifact_id", "sha256"):
        value = identity.get(key)
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("Completed real-stage identity/checksum is missing or invalid.")
    if method == "kae_moe" and "appended_kae" not in next_stage:
        raise ValueError("The completed real policy's fitted KAE is missing.")
    if method == "kae_moe":
        appended = next_stage["appended_kae"]
        record = next_stage.get("appended_kae_record", {})
        checksum = record.get("sha256", "")
        if (appended.get("kind") != "PortableKoopmanAutoencoder" or appended.get("padded_dim") != 57
                or appended.get("hidden_size") != 64 or not isinstance(checksum, str) or len(checksum) != 64
                or any(c not in "0123456789abcdef" for c in checksum)):
            raise ValueError("The appended KAE's architecture or verified record is missing.")
    if method == "fame" and (next_stage.get("consolidation_count") != 1 or "consolidated_fame_state" not in next_stage):
        raise ValueError("FAME requires the once-consolidated meta/task state.")
    if method == "fame":
        retained = next_stage["consolidated_fame_state"]
        if (retained.get("version") != 2 or retained.get("artifact_type") != "fame_meta_state"
                or retained.get("core_equations_version") != EQUATIONS[method]
                or retained.get("observation_contract") != OBSERVATION_CONTRACT
                or retained.get("shape") != manifest["shape"]):
            raise ValueError("Consolidated FAME meta/task contract mismatch.")
    return bundle


def _waypoint_actor_node(policy_state):
    state = {key: value for key, value in policy_state.items() if key.startswith("net_container.")}
    state.update({key.replace("policy_layer.0.", "policy_layer."): value for key, value in policy_state.items()
                  if key.startswith("policy_layer.0.")})
    first = nn.Linear(56, 64)
    second = nn.Linear(64, 64)
    output = nn.Linear(64, 3)
    return _node("BoundedDeterministicActor", state, input_dim=56, hidden_dims=[64, 64], output_dim=3,
                 networks={"hidden": [_layout(first), {"kind": "ELU", "alpha": 1.0}, _layout(second),
                                      {"kind": "ELU", "alpha": 1.0}],
                           "output": _layout(output), "output_transform": "tanh"})


def load_square_policy_checkpoint(path):
    data = Path(path).expanduser().resolve().read_bytes()
    import io
    payload = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
    if payload.get("observation_contract") != OBSERVATION_CONTRACT or "state_preprocessor" not in payload:
        raise ValueError("Square policy must contain the 56-input waypoint contract and normalizer.")
    state = payload["policy"]
    if (state["net_container.0.weight"].shape != (64, 56)
            or state["net_container.2.weight"].shape != (64, 64)
            or state["policy_layer.0.weight"].shape != (3, 64)
            or state["log_std_parameter"].shape != (3,)):
        raise ValueError("Square policy architecture is incompatible.")
    finite(payload)
    normalizer = payload["state_preprocessor"]
    if (normalizer["running_mean"].shape != (56,) or normalizer["running_variance"].shape != (56,)
            or (normalizer["running_variance"] < 0).any()
            or normalizer["current_count"].numel() != 1 or normalizer["current_count"].item() <= 0):
        raise ValueError("Square normalizer must preserve 56 finite statistics and its positive sample count.")
    payload["_source"] = {"path": str(Path(path).resolve()), "sha256": hashlib.sha256(data).hexdigest()}
    return payload


def load_square_controller(path, device="cpu"):
    payload = load_square_policy_checkpoint(path)
    state = payload["state_preprocessor"]
    normalizer = FrozenInputNormalizer(state)
    actor = _waypoint_actor_node(payload["policy"])
    node = _node("DeployablePolicy", {**_prefix(normalizer.state_dict(), "normalizer."),
                                      **_prefix(actor["state"], "actor.")},
                 normalizer=_normalizer_node(normalizer), actor=actor)
    return decode_actor(node, device).requires_grad_(False).eval()


class _LegacyModule(nn.Module):
    """Inert container for explicitly recognized historical square KAE pickles."""


class _SquareUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "Autoencoder" and name in ("KoopmanAutoencoder", "Encoder", "Decoder"):
            return _LegacyModule
        return super().find_class(module, name)


def build_square_kae_experts(path, device="cpu"):
    """Read the supplied full legacy KAE; never guess a missing Koopman matrix."""
    pickle_module = types.SimpleNamespace(Unpickler=_SquareUnpickler, load=pickle.load,
                                          __name__="warm_start_square_pickle")
    loaded = torch.load(path, map_location="cpu", weights_only=False, pickle_module=pickle_module)
    if isinstance(loaded, Mapping):
        loaded = loaded.get("kae", loaded.get("model", loaded))
    if not isinstance(loaded, nn.Module) or not hasattr(loaded, "K"):
        raise ValueError("Square KAE must contain the complete model and Koopman matrix K.")
    encoder = loaded.encoder.encoder if hasattr(loaded.encoder, "encoder") else loaded.encoder
    decoder = loaded.decoder.linear if hasattr(loaded.decoder, "linear") else loaded.decoder
    if encoder[0].in_features != 57 or encoder[0].out_features != 256:
        raise ValueError("Square KAE must use 57 inputs and hidden size 64.")
    matrix = loaded.K.detach().cpu().float()
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Square Koopman matrix must be square.")
    eigenvalues, vectors = torch.linalg.eig(matrix.to(torch.float64))
    inverse = torch.linalg.inv(vectors)
    state = {**_prefix(encoder.state_dict(), "encoder.encoder."), "decoder.linear.weight": decoder.weight,
             "K": matrix, "_captured_eigenvalues": eigenvalues,
             "_captured_eigenvectors": vectors, "_captured_eigenvectors_inverse": inverse}
    normalizer_path = Path(path).with_suffix(".state_preprocessor.pt")
    normalizer = nn.Identity()
    if normalizer_path.exists():
        normalizer = FrozenInputNormalizer(torch.load(normalizer_path, map_location="cpu", weights_only=True))
        state.update(_prefix(normalizer.state_dict(), "encoder.normalizer."))
    normalizer_node = _describe_sequence(normalizer) if isinstance(normalizer, nn.Identity) else _normalizer_node(normalizer)
    node = _node("PortableKoopmanAutoencoder", state, mode_count=matrix.shape[0], padded_dim=57, hidden_size=64,
                 normalizer=normalizer_node, networks={"encoder": _layout(encoder), "decoder": _layout(decoder)})
    finite(node)
    origin = "starting-square:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return [_expert_from_node(node, origin, device)]


def initialize_square_agent(agent, method, policy_checkpoint=None, kae_checkpoint=None):
    method = getattr(method, "name", method)
    if method not in ("finetune", "fame"):
        return {"source_policy_used_as_prior": True}
    payload = load_square_policy_checkpoint(policy_checkpoint)
    if method == "finetune":
        _load_waypoint_policy(agent.policy, payload["policy"])
        agent._state_preprocessor.load_state_dict(payload["state_preprocessor"])
        return {"source": payload["_source"], "preserved": ["actor", "noise", "observation_normalizer"],
                "resets": ["critic", "optimizer", "value_normalizer"]}
    native = _native("fame")
    configuration = {"hidden_dims": [64, 64], "normalization_state": payload["state_preprocessor"],
                     "normalizer_epsilon": 1e-8, "normalizer_clip_threshold": 5.0}
    actor = normalized_fame_actor(56, 3, [64, 64], configuration).to(agent.device)
    with torch.no_grad():
        state = payload["policy"]
        for target, source in ((0, 0), (2, 2)):
            actor.trunk[target].weight.copy_(state[f"net_container.{source}.weight"])
            actor.trunk[target].bias.copy_(state[f"net_container.{source}.bias"])
        actor.trunk[-1].weight.zero_()
        actor.trunk[-1].bias.zero_()
        actor.trunk[-1].weight[:3].copy_(state["policy_layer.0.weight"])
        actor.trunk[-1].bias[:3].copy_(state["policy_layer.0.bias"])
        mapped = 2 * (state["log_std_parameter"] - native.LOG_STD_MIN) / (native.LOG_STD_MAX - native.LOG_STD_MIN) - 1
        if (mapped.abs() >= 1).any():
            raise ValueError("Square exploration is outside FAME's invertible scale range.")
        actor.trunk[-1].bias[3:].copy_(torch.atanh(mapped))
    agent.hidden_dims = [64, 64]
    agent.cfg["hidden_dims"] = [64, 64]
    agent.fast_actor = actor
    agent.meta_actor = copy.deepcopy(actor)
    agent.fast_critic = _native("base").make_mlp(56, [64, 64], 1).to(agent.device)
    agent.fast_actor_opt = torch.optim.Adam(agent.fast_actor.parameters(), lr=agent.cfg["lr_actor"])
    agent.fast_critic_opt = torch.optim.Adam(agent.fast_critic.parameters(), lr=agent.cfg["lr_critic"])
    agent.checkpoint_modules.update(fast_actor=agent.fast_actor, fast_critic=agent.fast_critic, meta_actor=agent.meta_actor)
    agent._meta_trained = True
    return {"source": payload["_source"], "resets": ["critic", "optimizers", "temperature", "distillation_memory"],
            "fame_scale_head_seeded_from_source_noise": True}


def initialize_square_exploration(agent, policy_checkpoint):
    payload = load_square_policy_checkpoint(policy_checkpoint)
    with torch.no_grad():
        agent.policy.log_std_parameter.copy_(payload["policy"]["log_std_parameter"].to(agent.device))
    return {"source": payload["_source"], "noise": "preserved"}


def _load_waypoint_policy(policy, canonical):
    mapped = {"log_std_parameter": canonical["log_std_parameter"]}
    for native_prefix, canonical_prefix in (("actor.0.", "net_container.0."), ("actor.2.", "net_container.2."),
                                             ("actor.4.", "policy_layer.0.")):
        mapped.update({native_prefix + suffix: canonical[canonical_prefix + suffix] for suffix in ("weight", "bias")})
    policy.load_state_dict(mapped)


def _canonical_parameter_keys(state, role):
    return [role + "." + key for key in state if not key.startswith("policy_layer.")
            and (key == "log_std_parameter" or key.endswith((".weight", ".bias")))] + [
        role + "." + key for key in state if key.startswith("policy_layer.")
        and key.startswith("policy_layer.0.") and key.endswith((".weight", ".bias"))]


def _remap_adam(optimizer, native_names, target_names):
    if not isinstance(optimizer, torch.optim.Adam) or len(optimizer.param_groups) != 1:
        raise ValueError("Portable continuation supports one canonical Adam parameter group.")
    original = optimizer.state_dict()
    named_states = {}
    for identity, parameter in zip(original["param_groups"][0]["params"], optimizer.param_groups[0]["params"]):
        if id(parameter) not in native_names:
            if parameter.requires_grad:
                raise ValueError("Trainable optimizer parameter has no canonical mapping.")
            continue
        if identity in original["state"]:
            named_states[native_names[id(parameter)]] = original["state"][identity]
    if any(name not in target_names for name in named_states):
        raise ValueError("Optimizer state targets an unsupported canonical parameter.")
    group = {key: value for key, value in original["param_groups"][0].items() if key != "params"}
    group["params"] = list(range(len(target_names)))
    return plain({"state": {i: named_states[name] for i, name in enumerate(target_names) if name in named_states},
                  "param_groups": [group]})


def restore_finetune_agent(agent, bundle):
    modules = bundle["agent_modules"]
    _load_waypoint_policy(agent.policy, modules["policy"])
    agent.value.load_state_dict(modules["value"])
    for name in ("state_preprocessor", "value_preprocessor"):
        if name in modules:
            preprocessor = getattr(agent, "_" + name)
            if not hasattr(preprocessor, "load_state_dict"):
                raise ValueError("Native normalizer topology differs from the completed real PPO.")
            preprocessor.load_state_dict(modules[name])
    policy_node, policy_map = encode_current_policy("finetune", agent)
    native_names = {id(parameter): "policy." + policy_map[name] for name, parameter in agent.policy.named_parameters()}
    native_names.update({id(parameter): "value." + name for name, parameter in agent.value.named_parameters()})
    canonical_names = bundle["agent_configuration"]["optimizer_parameter_names"]["optimizer"][0]
    saved = modules["optimizer"]
    states = {name: saved["state"].get(identity) for name, identity in zip(canonical_names, saved["param_groups"][0]["params"])}
    current = agent.optimizer.state_dict()
    group = {key: value for key, value in saved["param_groups"][0].items() if key != "params"}
    group["params"] = current["param_groups"][0]["params"]
    remapped = {}
    for identity, parameter in zip(group["params"], agent.optimizer.param_groups[0]["params"]):
        name = native_names[id(parameter)]
        if name not in states:
            raise ValueError("Real PPO optimizer is missing a canonical parameter.")
        if states[name] is not None:
            remapped[identity] = states[name]
    agent.optimizer.load_state_dict({"state": remapped, "param_groups": [group]})
    scheduler = getattr(agent, "scheduler", None)
    if (scheduler is None) != (bundle["scheduler"] is None):
        raise ValueError("Native PPO scheduler topology differs from its completed real parent.")
    if scheduler is not None:
        scheduler.load_state_dict(bundle["scheduler"])
    if getattr(agent, "scaler", None) is not None and bundle["grad_scaler"] is not None:
        agent.scaler.load_state_dict(bundle["grad_scaler"])
    return {"preserved": ["actor", "critic", "normalizers", "noise", "optimizer", "scheduler"],
            "resets": ["on_policy_rollout", "backend_environment", "phase_counters"]}


def restore_fame_predecessor(agent, bundle):
    state = bundle["manifest"]["next_stage"]["consolidated_fame_state"]
    configuration = state["actor_configuration"]
    agent.hidden_dims = list(configuration["hidden_dims"])
    agent.cfg["hidden_dims"] = list(configuration["hidden_dims"])
    agent.meta_actor = normalized_fame_actor(56, 3, agent.hidden_dims, configuration).to(agent.device)
    agent.meta_actor.load_state_dict(state["meta_actor"])
    agent.checkpoint_modules["meta_actor"] = agent.meta_actor
    buffer_state = state["meta_buffer"]
    capacity = buffer_state["data"].shape[0]
    agent.meta_buffer = _native("fame").StateBuffer(capacity, 56, agent.device)
    agent.meta_buffer.data.copy_(buffer_state["data"].to(agent.device))
    agent.meta_buffer.index, agent.meta_buffer.size = int(buffer_state["index"]), int(buffer_state["size"])
    agent._meta_trained = bool(state["meta_trained"])
    agent.games, agent.flag_reg = list(state.get("games", [])), list(state.get("flag_reg", []))
    agent._portable_policy_initialization = plain(state.get("policy_initialization"))
    return {"preserved": ["once_consolidated_meta_actor", "meta_buffer", "task_history"],
            "consolidation_count": 1}


def _compo_node(unit, priors):
    state = {}
    for i, prior in enumerate(priors):
        state.update(_prefix(prior["state"], f"previous_units.{i}."))
    networks = {}
    for name in ("headout_wq", "headout_wk", "headin_wq", "headin_wk", "headin_wv", "internal_policy"):
        child = getattr(unit, name)
        state.update(_prefix(child.state_dict(), name + "."))
        networks[name] = _layout(child)
    state["pe1"] = unit.pe1
    state["pe0"] = unit.pe0 if unit.pe0 is not None else torch.zeros_like(unit.pe1[:, :-1, :])
    networks["attention_temperature"] = float(unit.att_temp)
    return _node("CompoNetCore", state, priors=priors, hidden_dim=unit.headout_wq.out_features,
                 equations=EQUATIONS["componet"], networks=networks)


def _describe_actor_and_map(actor):
    if hasattr(actor, "_portable_descriptor"):
        return plain(actor._portable_descriptor), dict(actor._portable_parameter_map)
    if hasattr(actor, "controller"):
        return _describe_actor_and_map(actor.controller)
    name = type(actor).__name__
    if isinstance(actor, (nn.Sequential, nn.Linear, nn.Identity, nn.Tanh, nn.ELU, nn.ReLU)):
        return _describe_sequence(actor), {key: key for key, _ in actor.named_parameters()}
    if isinstance(actor, FrozenInputNormalizer):
        return _normalizer_node(actor), {}
    if name == "DeterministicPolicy":
        child, mapping = _describe_actor_and_map(actor.actor)
        if actor.preprocessor is None:
            return child, {"actor." + k: v for k, v in mapping.items()}
        pre = actor.preprocessor
        normalizer = FrozenInputNormalizer(pre.state_dict(), getattr(pre, "epsilon", 1e-8), getattr(pre, "clip_threshold", 5.0))
        node = _node("DeployablePolicy", {**_prefix(normalizer.state_dict(), "normalizer."),
                                         **_prefix(child["state"], "actor.")},
                     actor=child, normalizer=_normalizer_node(normalizer))
        return node, {"actor." + k: "actor." + v for k, v in mapping.items()}
    if name == "ComposedActor":
        base, mapping = _describe_actor_and_map(actor.base)
        state = {**_prefix(base["state"], "base."), **_prefix(actor.residual.state_dict(), "residual_network.")}
        node = _node("FrozenBaseResidualCore", state, base=base,
                     chain_depth=getattr(actor, "_portable_chain_depth", getattr(actor.base, "_portable_chain_depth", 0) + 1),
                     equations=EQUATIONS["residual"], networks={"residual": _layout(actor.residual)})
        names = {"base." + k: "base." + v for k, v in mapping.items()}
        names.update({"residual." + k: "residual_network." + k for k, _ in actor.residual.named_parameters()})
        return node, names
    if name == "ActionOnly":
        unit = actor.unit
        if type(unit).__name__ == "FirstModuleWrapper":
            node, mapping = _describe_actor_and_map(unit.model)
            return node, {"unit.model." + key: value for key, value in mapping.items()}
        priors, names = [], {}
        for i, previous in enumerate(unit.previous_units):
            if hasattr(previous, "controller"):
                prior, mapping = _describe_actor_and_map(previous.controller)
                names.update({f"unit.previous_units.{i}.controller." + key: f"previous_units.{i}." + value
                              for key, value in mapping.items()})
            elif type(previous).__name__ == "FirstModuleWrapper":
                prior, mapping = _describe_actor_and_map(previous.model)
                names.update({f"unit.previous_units.{i}.model." + key: f"previous_units.{i}." + value
                              for key, value in mapping.items()})
            else:
                prior = _compo_node(previous, list(priors))
                names.update({f"unit.previous_units.{i}." + key: f"previous_units.{i}." + key
                              for key, _ in previous.named_parameters()})
            priors.append(prior)
        node = _compo_node(unit, priors)
        names.update({"unit." + key: key for key, _ in unit.named_parameters() if not key.startswith("previous_units.")})
        return node, names
    if name == "MoEActor":
        blocks = [plain(expert._portable_kae_descriptor) for expert in actor.kaes]
        origins = [expert._portable_origin for expert in actor.kaes]
        offset, slices, bank_state = 0, [], {}
        names = {}
        for i, (block, expert) in enumerate(zip(blocks, actor.kaes)):
            slices.append([offset, offset + block["mode_count"]])
            offset += block["mode_count"]
            bank_state.update(_prefix(block["state"], f"blocks.{i}."))
            names.update({f"kaes.{i}." + key: f"kae.blocks.{i}." + value
                          for key, value in expert._portable_parameter_map.items()})
        bank = _node("StackedPortableKAE", bank_state, blocks=blocks, origins=origins, mode_slices=slices)
        state = _prefix(bank_state, "kae.")
        for name in ("mlp_network", "expert_weight_networks", "kae_router_network", "gating_network"):
            state.update(_prefix(getattr(actor, name).state_dict(), "head." + name + "."))
        names.update({key: "head." + key for key, _ in actor.named_parameters() if not key.startswith("kaes.")})
        networks = {"mlp": _layout(actor.mlp_network), "mode_weight_networks": [_layout(net) for net in actor.expert_weight_networks],
                    "router": _layout(actor.kae_router_network), "gate": _layout(actor.gating_network),
                    "branch_mode_counts": [block["mode_count"] for block in blocks], "router_activation": "softmax",
                    "mode_weight_normalization": "signed_l1_epsilon_1e-6", "gate_activation": "sigmoid"}
        return _node("ExpandingKaeMoECore", state, kae=bank, steps=1, pad_value=1., residual_scale=1.,
                     sensitivity_scale=0., step_dt=.02, equations=EQUATIONS["kae_moe"], networks=networks), names
    if name in ("SquashedActor", "ImportedFameActor"):
        configuration = getattr(actor, "_portable_fame_configuration", None)
        if configuration is None:
            configuration = {"hidden_dims": [layer.out_features for layer in actor.trunk if isinstance(layer, nn.Linear)][:-1],
                             "normalization_state": None, "normalizer_epsilon": 1e-8, "normalizer_clip_threshold": 5.0}
        else:
            configuration = plain(configuration)
            configuration.setdefault("normalizer_epsilon", 1e-8)
            configuration.setdefault("normalizer_clip_threshold", 5.0)
            if not isinstance(actor.input_normalizer, nn.Identity):
                configuration["normalization_state"] = plain(actor.input_normalizer.state_dict())
        return _node("FameActor", actor.state_dict(), configuration=configuration,
                     equations=EQUATIONS["fame"], networks={"trunk": _layout(actor.trunk)}), {
                         key: key for key, _ in actor.named_parameters()}
    if name == "SquashedMean":
        child, mapping = _describe_actor_and_map(actor.actor)
        return _node("SquashedMean", _prefix(child["state"], "actor."), actor=child), {
            "actor." + key: "actor." + value for key, value in mapping.items()}
    raise ValueError(f"Unsupported native actor architecture: {name}.")


def describe_native_actor(actor):
    return _describe_actor_and_map(actor)[0]


def encode_current_policy(method, agent):
    method = getattr(method, "name", method)
    if method == "fame":
        return _describe_actor_and_map(agent.fast_actor)
    if not hasattr(agent.policy, "_min_log_std") or agent.policy._min_log_std != -3.0:
        raise ValueError("Original waypoint continuation requires the -3 learned-noise floor.")
    if method == "finetune":
        actor = agent.policy.actor
        if not isinstance(actor, nn.Sequential) or _layout(actor) != [
            {"kind": "Linear", "in_features": 56, "out_features": 64, "bias": True}, {"kind": "ELU", "alpha": 1.0},
            {"kind": "Linear", "in_features": 64, "out_features": 64, "bias": True}, {"kind": "ELU", "alpha": 1.0},
            {"kind": "Linear", "in_features": 64, "out_features": 3, "bias": True}, {"kind": "Tanh"}]:
            raise ValueError("PPO continuation requires the original 56→64→64→3 waypoint actor.")
        state = {"log_std_parameter": agent.policy.log_std_parameter}
        mapping = {"log_std_parameter": "log_std_parameter"}
        for native, canonical in (("0.", "net_container.0."), ("2.", "net_container.2."), ("4.", "policy_layer.0.")):
            for suffix in ("weight", "bias"):
                state[canonical + suffix] = actor.state_dict()[native + suffix]
                mapping["actor." + native + suffix] = canonical + suffix
        return _node("CrazySimWaypointPolicy", state, min_log_std=-3.0, equations=EQUATIONS[method],
                     networks={"hidden": [_layout(layer) for layer in list(actor)[:4]],
                               "output": [_layout(actor[4]), _layout(actor[5])], "output_transform": "as_declared"}), mapping
    core, names = _describe_actor_and_map(agent.policy.actor)
    kind, prefix = {"residual": ("ResidualPolicy", "residual_core."), "componet": ("CompoNetPolicy", "componet_core."),
                    "kae_moe": ("ExpandingKaeMoEPolicy", "moe_core.")}[method]
    expected_core = {"residual": "FrozenBaseResidualCore", "componet": "CompoNetCore", "kae_moe": "ExpandingKaeMoECore"}[method]
    if core["kind"] != expected_core:
        raise ValueError("Initial Square preparation must construct the method-specific learner before export.")
    state = {"log_std_parameter": agent.policy.log_std_parameter,
             **_prefix(core["state"], prefix), **_prefix(core["state"], "policy_layer.")}
    fields = {key: value for key, value in core.items() if key not in ("kind", "state", "networks", "equations")}
    if method == "residual":
        fields["policy_initialization"] = "legacy"
    node = _node(kind, state, min_log_std=-3.0, equations=EQUATIONS[method], networks=core["networks"], **fields)
    return node, {"log_std_parameter": "log_std_parameter",
                  **{"actor." + key: prefix + value for key, value in names.items()}}


def _preprocessor_info(preprocessor):
    if not hasattr(preprocessor, "state_dict"):
        return None
    return {"kind": "SingleDroneRunningStandardScaler", "size": preprocessor.running_mean.numel(),
            "epsilon": float(preprocessor.epsilon), "clip_threshold": float(preprocessor.clip_threshold)}


def _buffer_state(buffer):
    return {"data": plain(buffer.data), "index": int(buffer.index), "size": int(buffer.size)}


def export_warmed_learner(method, agent, output_path, *, parent, shape, environment, iterations,
                          counters, checkpoint_selection="exact_cutoff", source_revision="unknown", initialization=None):
    """Export the final current learner without stage finalization or rollout data."""
    method_name = getattr(method, "name", method)
    if method_name not in EQUATIONS or shape not in SHAPES or type(iterations) is not int or iterations <= 0:
        raise ValueError("Warm export requires a supported method/shape and positive iteration budget.")
    if checkpoint_selection != "exact_cutoff":
        raise ValueError("Only exact-cutoff state is exported; arbitrary best-policy files lack compatible method state.")
    if (type(counters.get("complete_updates")) is not int or counters["complete_updates"] != iterations
            or type(counters.get("accepted_transitions")) is not int or counters["accepted_transitions"] <= 0):
        raise ValueError("Exact-cutoff export requires observed complete-update and accepted-transition counters.")
    if environment.get("shape") != shape:
        raise ValueError("Export environment shape differs from its learner.")
    policy, parameter_map = encode_current_policy(method_name, agent)
    if method_name == "fame":
        value = _node("FameValue", _prefix(agent.fast_critic.state_dict(), "net."), hidden_dims=list(agent.hidden_dims),
                      networks={"value": _layout(agent.fast_critic)})
        policy_names = _canonical_parameter_keys(policy["state"], "policy")
        value_names = _canonical_parameter_keys(value["state"], "value")
        meta, _ = _describe_actor_and_map(agent.meta_actor)
        native_policy_names = {id(parameter): "policy." + name for name, parameter in agent.fast_actor.named_parameters()}
        native_value_names = {id(parameter): "value.net." + name for name, parameter in agent.fast_critic.named_parameters()}
        buffers = {"version": 2, "core_equations_version": EQUATIONS[method_name],
                   "fast2meta": _buffer_state(agent.fast2meta), "meta": _buffer_state(agent.meta_buffer),
                   "meta_trained": bool(agent._meta_trained), "games": list(getattr(agent, "games", [])),
                   "flag_reg": list(getattr(agent, "flag_reg", [])), "actor_configuration": policy["configuration"],
                   "policy_initialization": plain(getattr(agent, "_portable_policy_initialization", None))}
        modules = {"policy": policy["state"], "value": value["state"],
                   "optimizer": _remap_adam(agent.fast_actor_opt, native_policy_names, policy_names),
                   "critic_optimizer": _remap_adam(agent.fast_critic_opt, native_value_names, value_names),
                   "temperature": {"log_alpha": plain(agent.log_alpha)},
                   "temperature_optimizer": _remap_adam(agent.log_alpha_opt, {id(agent.log_alpha): "temperature.log_alpha"}, ["temperature.log_alpha"]),
                   "meta_actor": meta["state"], "fame_buffers": buffers}
        optimizer_names = {"optimizer": [policy_names], "critic_optimizer": [value_names],
                           "temperature_optimizer": [["temperature.log_alpha"]]}
        settings = {"rollouts": agent.cfg["rollout_steps"], "mixed_precision": False,
                    "fame_size_fast2meta": agent.fast2meta.capacity, "fame_size_meta": agent.meta_buffer.capacity,
                    "fame_observation_contract": OBSERVATION_CONTRACT,
                    "fame_core": {key: plain(value) for key, value in agent.cfg.items() if key != "experiment"},
                    "rewards_shaper_scale": agent.cfg["rewards_shaper_scale"]}
        preprocessors = {"state_preprocessor": None, "value_preprocessor": None}
        scheduler_info, scheduler_state, scaler_state = None, None, {}
    else:
        value = _node("BoundedValue", agent.value.state_dict(), hidden_dims=[64, 64],
                      networks={"value": _layout(agent.value.net)})
        canonical_names = _canonical_parameter_keys(policy["state"], "policy") + _canonical_parameter_keys(value["state"], "value")
        native_names = {id(parameter): "policy." + parameter_map[name] for name, parameter in agent.policy.named_parameters()
                        if name in parameter_map}
        native_names.update({id(parameter): "value." + name for name, parameter in agent.value.named_parameters()})
        modules = {"policy": policy["state"], "value": value["state"],
                   "optimizer": _remap_adam(agent.optimizer, native_names, canonical_names)}
        preprocessors = {}
        for key in ("state_preprocessor", "value_preprocessor"):
            pre = getattr(agent, "_" + key)
            preprocessors[key] = _preprocessor_info(pre)
            if preprocessors[key] is not None:
                modules[key] = plain(pre.state_dict())
        settings = {}
        excluded = {"experiment", "state_preprocessor", "state_preprocessor_kwargs", "value_preprocessor",
                    "value_preprocessor_kwargs", "learning_rate_scheduler", "learning_rate_scheduler_kwargs", "rewards_shaper"}
        for key, item in agent.cfg.items():
            if key not in excluded:
                settings[key] = plain(item)
        scheduler = getattr(agent, "scheduler", None)
        if scheduler is not None and not isinstance(scheduler, torch.optim.lr_scheduler.ExponentialLR):
            raise ValueError("The dedicated warm-up path must configure a compatible ExponentialLR or no scheduler.")
        scheduler_info = None if scheduler is None else {"kind": "ExponentialLR", "gamma": scheduler.gamma}
        scheduler_state = None if scheduler is None else plain(scheduler.state_dict())
        scaler_state = None if getattr(agent, "scaler", None) is None else plain(agent.scaler.state_dict())
        optimizer_names = {"optimizer": [canonical_names]}
    parent_record = None
    software_validation_only = False
    cumulative = plain(counters)
    if parent is not None:
        pm = parent["manifest"]
        if pm["method"] != method_name:
            raise ValueError("A warmed learner cannot change method histories.")
        parent_record = {"artifact_id": pm["artifact_id"], "method": method_name, "shape": pm["shape"], "phase": pm["phase"],
                         **plain(parent.get("_source", {}))}
        software_validation_only = bool(pm["phase"] == "kinematic" or pm.get("software_validation_only")
                                        or (pm.get("next_stage") or {}).get("software_validation_only"))
        for key, amount in pm["cumulative_counters"].items():
            if isinstance(amount, (int, float)):
                cumulative[key] = cumulative.get(key, 0) + amount
    resets = ["on_policy_rollout", "backend_environment", "backend_rng", "phase_counters"]
    environment = plain(environment)
    num_envs = getattr(getattr(method, "env", None), "num_envs", None)
    if num_envs is not None and counters["accepted_transitions"] != iterations * settings["rollouts"] * num_envs:
        raise ValueError("Observed accepted transitions do not match completed native rollout updates.")
    manifest = {"version": 1, "artifact_type": "simulation_real_continuation", "method": method_name,
                "equations": EQUATIONS[method_name], "source_revision": source_revision,
                "source_dirty_diff_sha256": None, "source_adapter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "shape": shape, "phase": "simulation", "stage_identity": f"{method_name}:{shape}:simulation:{Path(output_path).parent.name}",
                "software_validation_only": software_validation_only,
                "parent": parent_record, "dependencies": dependencies(policy), "observation_contract": OBSERVATION_CONTRACT,
                "observation_dim": 56, "action_dim": 3, "observation_specification": OBSERVATION_SPECIFICATION,
                "action_contract": {"bounds": [-1., 1.], "decoder": "position_plus_scaled_action",
                                    "offset_limit_m": environment["waypoint_offset_limit"], "yaw_radians": 0.0},
                "distribution": distribution(method_name), "environment": environment,
                "phase_counters": plain(counters), "counters": plain(counters), "cumulative_counters": cumulative,
                "budget": {"max_episodes": None, "accepted_transitions": counters["accepted_transitions"],
                           "complete_updates": iterations, "allocated_iterations": iterations,
                           "num_envs": num_envs,
                           "rollout_steps": settings["rollouts"]},
                "checkpoint_selection": {"rule": checkpoint_selection, "actual_update": counters["complete_updates"], "actual_iteration": iterations},
                "resets": list(dict.fromkeys(resets)), "initialization": plain(initialization),
                "new_shape_initialization_resets": ([] if initialization is None else
                                                    plain(initialization.get("state_resets", initialization.get("resets", [])))),
                "preserved_method_memory": ["fast2meta", "meta"] if method_name == "fame" else [], "next_stage": None}
    bundle = {"manifest": manifest, "architecture": {"policy": policy, "value": value}, "agent_modules": modules,
              "agent_configuration": {"settings": settings, "preprocessors": preprocessors,
                                      "scheduler": scheduler_info, "optimizer_types": {key: "Adam" for key in optimizer_names},
                                      "optimizer_parameter_names": optimizer_names},
              "scheduler": scheduler_state, "grad_scaler": scaler_state}
    return write_bundle(bundle, output_path)


def validate_square_inputs(policy_path, kae_path=None):
    payload = load_square_policy_checkpoint(policy_path)
    result = {"policy": payload["_source"], "observation_contract": OBSERVATION_CONTRACT,
              "actor_hidden_dims": [64, 64], "action_dim": 3,
              "normalizer_count": float(payload["state_preprocessor"]["current_count"])}
    if kae_path is not None:
        # The legacy pickle compatibility loader is independent of Isaac; only
        # constructing native experts requires the simulator method namespace.
        pickle_module = types.SimpleNamespace(Unpickler=_SquareUnpickler, load=pickle.load,
                                              __name__="warm_start_square_pickle")
        model = torch.load(kae_path, map_location="cpu", weights_only=False, pickle_module=pickle_module)
        if not isinstance(model, nn.Module) or not hasattr(model, "K"):
            raise ValueError("Square KAE must retain the full model and Koopman matrix.")
        encoder = model.encoder.encoder if hasattr(model.encoder, "encoder") else model.encoder
        decoder = model.decoder.linear if hasattr(model.decoder, "linear") else model.decoder
        matrix = model.K
        if (encoder[0].in_features != 57 or encoder[0].out_features != 256 or matrix.ndim != 2
                or matrix.shape[0] != matrix.shape[1] or decoder.weight.shape != (57, matrix.shape[0])):
            raise ValueError("Square KAE architecture mismatch.")
        finite(model.state_dict())
        finite(matrix)
        result["kae"] = {"path": str(Path(kae_path).resolve()), "sha256": hashlib.sha256(Path(kae_path).read_bytes()).hexdigest(),
                         "mode_count": matrix.shape[0], "padded_dim": 57}
    return result
