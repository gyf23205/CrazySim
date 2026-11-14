"""Simulation warm-up of one shape and its handoff to ``Drone_Test_TRO`` (see ``PLAN.md``).

A warm-up trains one method on one new shape for a fixed number of iterations, keeps the best checkpoint
and writes it as a ``transfer_bundle.pt`` (:mod:`crazysim.transfer_bundle`). The real drone continues
that learner; its finalized result is the parent of the next shape's warm-up. This module holds what
``scripts/skrl/train.py`` needs for that:

- :func:`seed_from_square` and :func:`seed_from_bundle` give a method the state of its parent, as
  ``Method.resume`` does for a finished stage of this repository.
- :func:`finish_warm_stage` replaces ``Method.end_stage``: it selects the best checkpoint and exports it,
  without the steps that finish a shape (KAE fit, FAME consolidation). Those happen after the real flight.

The method classes are used unchanged. Parameter names differ between the two repositories, so a learner
is written as a *description*: a nested mapping with a ``kind``, a state dictionary with the names of
``Drone_Test_TRO``, and the settings needed to rebuild it. Modules rebuilt from a description remember it
(``_bundle_node``), so that a frozen earlier policy is written back exactly as it was read.
"""

from __future__ import annotations

import copy
import hashlib
import os
import subprocess
import torch
import torch.nn as nn
from collections.abc import Mapping
from pathlib import Path

from . import transfer_bundle as tb
from .methods.base import DeterministicPolicy, Method, freeze, make_mlp
from .methods.componet import ActionOnly, CompoNet
from .methods.fame import SquashedActor, SquashedMean, StateBuffer
from .methods.kae import KAEExperts, KoopmanAutoencoder
from .methods.kae_moe import MoEActor
from .methods.residual import ComposedActor

SHARED_SQUARE_METHODS = ("finetune", "residual", "componet", "kae_moe")
"""Methods that can start from a Square trained with ``--method scratch``: their own first stage is the same
plain PPO policy. FAME trains its own Square."""
SQUARE_FILES = {
    "finetune": ("checkpoint",),
    "residual": ("checkpoint",),
    "componet": ("checkpoint",),
    "kae_moe": ("kae",),
    "fame": ("policy", "meta_policy"),
}
"""Files of the Square stage a warm-up of each method starts from (arguments of :func:`seed_from_square`)."""
NEXT_STAGE_STRATEGY = {
    "finetune": "continue_actor_and_compatible_algorithm_state",
    "residual": "freeze_completed_current_controller_as_base",
    "componet": "append_completed_current_controller_to_prior_stack",
    "kae_moe": "append_full_real_policy_kae",
    "fame": "retain_once_consolidated_meta_state",
}
"""What ``finalize_stage.py`` of ``Drone_Test_TRO`` records for the next shape, per method."""
OBS_DIM, ACTION_DIM = 56, 3


# -- modules inside rebuilt policies -------------------------------------------------------------------


class FrozenNormalizer(nn.Module):
    """Observation normalization with fixed statistics; the formula of skrl's ``RunningStandardScaler``."""

    def __init__(self, state: Mapping, epsilon: float = 1e-8, clip_threshold: float = 5.0):
        super().__init__()
        self.register_buffer("running_mean", state["running_mean"].detach().float().clone())
        self.register_buffer("running_variance", state["running_variance"].detach().float().clone())
        if "current_count" in state:
            self.register_buffer("current_count", state["current_count"].detach().clone())
        self.epsilon, self.clip_threshold = float(epsilon), float(clip_threshold)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        scaled = (obs.float() - self.running_mean) / (self.running_variance.sqrt() + self.epsilon)
        return scaled.clamp(-self.clip_threshold, self.clip_threshold)


class NormalizedKaeEncoder(nn.Module):
    """KAE encoder that normalizes the observation part of its input and leaves the padding as it is."""

    def __init__(self, network: nn.Module, normalizer: nn.Module):
        super().__init__()
        self.network, self.normalizer = network, normalizer

    def forward(self, padded_obs: torch.Tensor) -> torch.Tensor:
        obs = self.normalizer(padded_obs[..., :OBS_DIM])
        return self.network(torch.cat((obs, padded_obs[..., OBS_DIM:]), dim=-1))


class PriorUnit(nn.Module):
    """A complete frozen controller as an earlier unit of :class:`~crazysim.methods.componet.CompoNet`.

    The units of this repository pass the actions of all earlier units along a chain. A controller from
    ``Drone_Test_TRO`` computes its own action from the observation alone, so it only appends that action.
    """

    def __init__(self, controller: nn.Module):
        super().__init__()
        self.controller = controller
        self.is_prev = False

    def forward(self, value):
        phi, obs = value if isinstance(value, tuple) else (None, value)
        action = self.controller(obs)
        phi = action[:, None, :] if phi is None else torch.cat((phi, action[:, None, :]), dim=1)
        return (phi, obs) if self.is_prev else (action, phi)


# -- descriptions ----------------------------------------------------------------------------------------


def _prefix(state: Mapping, prefix: str) -> dict:
    return {prefix + key: value for key, value in state.items()}


def _strip(state: Mapping, prefix: str) -> dict:
    return {key[len(prefix) :]: value for key, value in state.items() if key.startswith(prefix)}


def _layout(module: nn.Module):
    """Layer sizes and activations of a network, as ``Drone_Test_TRO`` records them."""
    if isinstance(module, nn.Sequential):
        return [_layout(child) for child in module]
    if isinstance(module, nn.Linear):
        return {
            "kind": "Linear",
            "in_features": module.in_features,
            "out_features": module.out_features,
            "bias": module.bias is not None,
        }
    if isinstance(module, nn.ELU):
        return {"kind": "ELU", "alpha": float(module.alpha)}
    if isinstance(module, (nn.Tanh, nn.ReLU, nn.Identity)):
        return {"kind": type(module).__name__}
    raise ValueError(f"Unsupported network layer {type(module).__name__}.")


def _hidden_dims(layout: list) -> list[int]:
    """Hidden layer sizes of a recorded MLP layout."""
    return [layer["out_features"] for layer in layout if isinstance(layer, Mapping) and layer["kind"] == "Linear"][:-1]


def _node(kind: str, state: Mapping | None = None, *, networks=None, **fields) -> dict:
    return {"kind": kind, "state": tb.plain(state or {}), "networks": networks or {}, **tb.plain(fields)}


def _remember(module: nn.Module, node: Mapping, names: Mapping | None = None) -> nn.Module:
    """Attach the description of ``module`` and the map from its parameter names to the names in the description."""
    module._bundle_node = tb.plain(node)
    module._bundle_names = dict(names) if names is not None else {name: name for name, _ in module.named_parameters()}
    return module


def _parameter_names(state: Mapping, role: str) -> list[str]:
    """Parameters of a description in the order of the optimizer of ``Drone_Test_TRO``.

    ``policy_layer.`` repeats the trainable core of a composed policy, so those entries are skipped; the plain
    waypoint policy is the exception, whose output layer is ``policy_layer.0``.
    """
    is_parameter = lambda key: key == "log_std_parameter" or key.endswith((".weight", ".bias"))  # noqa: E731
    names = [f"{role}.{key}" for key in state if not key.startswith("policy_layer.") and is_parameter(key)]
    return names + [f"{role}.{key}" for key in state if key.startswith("policy_layer.0.") and is_parameter(key)]


def controller_from_node(node: Mapping, device="cpu") -> nn.Module:
    """Rebuild a policy of ``Drone_Test_TRO`` from its description, with the classes of this repository."""
    kind, state = node["kind"], node["state"]
    names = None
    if kind == "Identity":
        module = nn.Identity()
    elif kind in ("RunningStandardScalerModule", "SingleDroneRunningStandardScaler"):
        module = FrozenNormalizer(state, node["epsilon"], node["clip_threshold"])
    elif kind in ("BoundedDeterministicActor", "DeterministicActor", "CrazySimWaypointPolicy"):
        # 56 -> 64 -> 64 -> 3 MLP; the waypoint PPO policy keeps its output layer in a Sequential with the tanh
        output = "policy_layer.0." if kind == "CrazySimWaypointPolicy" else "policy_layer."
        hidden = [state["net_container.0.weight"].shape[0], state["net_container.2.weight"].shape[0]]
        module = make_mlp(OBS_DIM, hidden, ACTION_DIM)
        if kind != "DeterministicActor":
            module.append(nn.Tanh())
        names = {}
        for native, canonical in (("0.", "net_container.0."), ("2.", "net_container.2."), ("4.", output)):
            names.update({native + suffix: canonical + suffix for suffix in ("weight", "bias")})
        module.load_state_dict({native: state[canonical] for native, canonical in names.items()})
    elif kind == "DeployablePolicy":
        actor = controller_from_node(node["actor"], device)
        module = DeterministicPolicy(actor, controller_from_node(node["normalizer"], device))
        names = {"actor." + key: "actor." + value for key, value in actor._bundle_names.items()}
    elif kind in ("ResidualPolicy", "FrozenBaseResidualCore"):
        core = _strip(state, "residual_core.") if kind == "ResidualPolicy" else state
        base = controller_from_node(node["base"], device)
        residual = make_mlp(OBS_DIM, _hidden_dims(node["networks"]["residual"]), ACTION_DIM)
        residual.load_state_dict(_strip(core, "residual_network."))
        module = ComposedActor(base, residual)
        module._chain_depth = node["chain_depth"]
        prefix = "residual_core." if kind == "ResidualPolicy" else ""
        names = {"base." + key: prefix + "base." + value for key, value in base._bundle_names.items()}
        names.update({"residual." + key: prefix + "residual_network." + key for key, _ in residual.named_parameters()})
    elif kind in ("CompoNetPolicy", "CompoNetCore"):
        core = _strip(state, "componet_core.") if kind == "CompoNetPolicy" else state
        priors = [controller_from_node(prior, device) for prior in node["priors"]]
        hidden_dim = node["hidden_dim"]
        internal = make_mlp(OBS_DIM + hidden_dim, _hidden_dims(node["networks"]["internal_policy"]), ACTION_DIM)
        unit = CompoNet([PriorUnit(prior) for prior in priors], OBS_DIM, hidden_dim, ACTION_DIM, internal, device)
        own = {key: value for key, value in core.items() if not key.startswith("previous_units.")}
        if unit.pe0 is None:
            own.pop("pe0", None)
        result = unit.load_state_dict(own, strict=False)
        if result.unexpected_keys or any(not key.startswith("previous_units.") for key in result.missing_keys):
            raise ValueError("The CompoNet description does not match the attention heads and internal policy.")
        module = ActionOnly(unit)
        prefix = "componet_core." if kind == "CompoNetPolicy" else ""
        names = {
            "unit." + key: prefix + key for key, _ in unit.named_parameters() if not key.startswith("previous_units.")
        }
        for index, prior in enumerate(priors):
            for key, value in prior._bundle_names.items():
                names[f"unit.previous_units.{index}.controller.{key}"] = f"{prefix}previous_units.{index}.{value}"
    elif kind in ("ExpandingKaeMoEPolicy", "ExpandingKaeMoECore"):
        core = _strip(state, "moe_core.") if kind == "ExpandingKaeMoEPolicy" else state
        bank, networks = node["kae"], node["networks"]
        experts = [
            kae_expert_from_node(block, origin, device) for block, origin in zip(bank["blocks"], bank["origins"])
        ]
        cfg = {
            "residual_hidden_dims": _hidden_dims(networks["mlp"]),
            "weight_hidden_dims": _hidden_dims(networks["mode_weight_networks"][0]),
            "router_hidden_dims": _hidden_dims(networks["router"]),
            "gating_hidden_dims": _hidden_dims(networks["gate"]),
        }
        module = MoEActor(experts, OBS_DIM, ACTION_DIM, bank["blocks"][0]["padded_dim"], cfg)
        result = module.load_state_dict(_strip(core, "head."), strict=False)
        if result.unexpected_keys or any(not key.startswith("kaes.") for key in result.missing_keys):
            raise ValueError("The KAE-MoE description does not match the router, gate, mode-weight and MLP networks.")
        prefix = "moe_core." if kind == "ExpandingKaeMoEPolicy" else ""
        names = {key: prefix + "head." + key for key, _ in module.named_parameters() if not key.startswith("kaes.")}
        for index, expert in enumerate(experts):
            for key, value in expert._bundle_names.items():
                names[f"kaes.{index}.{key}"] = f"{prefix}kae.blocks.{index}.{value}"
    elif kind == "FameActor":
        configuration = node["configuration"]
        if configuration.get("normalization_state") is not None:
            raise ValueError("A FAME actor with an input normalizer is not supported: FAME here uses raw observations.")
        module = SquashedActor(OBS_DIM, ACTION_DIM, list(configuration["hidden_dims"]))
        module.load_state_dict(state)
    elif kind == "SquashedMean":
        actor = controller_from_node(node["actor"], device)
        module = SquashedMean(actor)
        names = {"actor." + key: "actor." + value for key, value in actor._bundle_names.items()}
    else:
        raise ValueError(f"Unsupported policy description of kind {kind!r}.")
    return _remember(module.to(device), node, names)


def kae_expert_from_node(node: Mapping, origin: str, device="cpu") -> KAEExperts:
    """Experts of one KAE of ``Drone_Test_TRO``, with the eigen-modes in the recorded order.

    The mode-weight networks of a KAE-MoE policy are tied to the order of the modes, so the recorded
    eigen-decomposition is used instead of a new one.
    """
    state = node["state"]
    kae = KoopmanAutoencoder(node["padded_dim"], node["hidden_size"], node["mode_count"], device)
    kae.encoder.load_state_dict(_strip(state, "encoder.encoder."))
    encoder_names = {"encoder." + key: "encoder.encoder." + key for key, _ in kae.encoder.named_parameters()}
    normalizer = controller_from_node(node["normalizer"], device)
    if not isinstance(normalizer, nn.Identity):
        encoder_names = {
            "encoder.network." + key: "encoder.encoder." + key for key, _ in kae.encoder.named_parameters()
        }
        kae.encoder = NormalizedKaeEncoder(kae.encoder, normalizer)
    kae.decoder.load_state_dict({"weight": state["decoder.linear.weight"]})
    kae.K = state["K"].to(device)
    expert = KAEExperts(kae.to(device), ACTION_DIM, p=1).to(device)
    # action = decoder (V diag(eigenvalues) V^-1) z, one term per eigenvalue
    decoder = state["decoder.linear.weight"][:ACTION_DIM].to(torch.complex128)
    expert.eigvals_p = state["_captured_eigenvalues"].to(device=device, dtype=torch.complex64)
    expert.eigvec_left = state["_captured_eigenvectors_inverse"].to(device=device, dtype=torch.complex64)
    expert.v = (decoder @ state["_captured_eigenvectors"]).to(device=device, dtype=torch.complex64)
    expert._bundle_origin = str(origin)
    return _remember(freeze(expert), node, encoder_names)


def kae_node(kae: KoopmanAutoencoder) -> dict:
    """Description of a KAE fitted in this repository (``kae/kae.pt`` of a stage)."""
    matrix = kae.K.detach().cpu().float()
    eigenvalues, eigenvectors = torch.linalg.eig(matrix.to(torch.float64))
    state = {
        "K": matrix,
        "_captured_eigenvalues": eigenvalues,
        "_captured_eigenvectors": eigenvectors,
        "_captured_eigenvectors_inverse": torch.linalg.inv(eigenvectors),
        **_prefix(kae.encoder.state_dict(), "encoder.encoder."),
        "decoder.linear.weight": kae.decoder.weight,
    }
    return _node(
        "PortableKoopmanAutoencoder",
        state,
        mode_count=matrix.shape[0],
        padded_dim=kae.encoder[0].in_features,
        hidden_size=kae.encoder[0].out_features // 4,
        normalizer=_node("Identity"),
        networks={"encoder": _layout(kae.encoder), "decoder": _layout(kae.decoder)},
    )


def ppo_controller_node(checkpoint: Mapping) -> dict:
    """Description of the deterministic policy in a PPO agent checkpoint of this repository (plain MLP actor)."""
    policy = checkpoint["policy"]
    state = {}
    for native, canonical in (
        ("actor.0.", "net_container.0."),
        ("actor.2.", "net_container.2."),
        ("actor.4.", "policy_layer."),
    ):
        state.update({canonical + suffix: policy[native + suffix] for suffix in ("weight", "bias")})
    hidden = [state["net_container.0.weight"].shape[0], state["net_container.2.weight"].shape[0]]
    layers = make_mlp(OBS_DIM, hidden, ACTION_DIM)
    actor = _node(
        "BoundedDeterministicActor",
        state,
        input_dim=OBS_DIM,
        hidden_dims=hidden,
        output_dim=ACTION_DIM,
        networks={"hidden": _layout(layers[:4]), "output": _layout(layers[4]), "output_transform": "tanh"},
    )
    preprocessor = checkpoint.get("state_preprocessor")
    if preprocessor is None:
        return actor
    normalizer = FrozenNormalizer(preprocessor)
    normalizer_node = _node(
        "RunningStandardScalerModule",
        normalizer.state_dict(),
        epsilon=normalizer.epsilon,
        clip_threshold=normalizer.clip_threshold,
    )
    state = {**_prefix(normalizer_node["state"], "normalizer."), **_prefix(actor["state"], "actor.")}
    return _node("DeployablePolicy", state, normalizer=normalizer_node, actor=actor)


def describe_controller(module: nn.Module) -> tuple[dict, dict]:
    """Description of a policy network of this repository and the map from its parameter names to the description's.

    Frozen parts must have been rebuilt from a description (:func:`controller_from_node`); only the networks
    that a new stage adds on top of them are described here.
    """
    if hasattr(module, "_bundle_node"):
        return tb.plain(module._bundle_node), dict(module._bundle_names)
    if isinstance(module, PriorUnit):
        return describe_controller(module.controller)
    if isinstance(module, ComposedActor):
        base, base_names = describe_controller(module.base)
        state = {**_prefix(base["state"], "base."), **_prefix(module.residual.state_dict(), "residual_network.")}
        node = _node(
            "FrozenBaseResidualCore",
            state,
            base=base,
            chain_depth=getattr(module.base, "_chain_depth", 0) + 1,
            equations=tb.EQUATIONS["residual"],
            networks={"residual": _layout(module.residual)},
        )
        names = {"base." + key: "base." + value for key, value in base_names.items()}
        names.update({"residual." + key: "residual_network." + key for key, _ in module.residual.named_parameters()})
        return node, names
    if isinstance(module, ActionOnly) and isinstance(module.unit, CompoNet):
        unit = module.unit
        priors, names = [], {}
        # the order of the entries is the order of the parameters in the optimizer of Drone_Test_TRO
        state = {
            "pe1": unit.pe1,
            "pe0": unit.pe0 if unit.pe0 is not None else torch.zeros_like(unit.pe1[:, :-1, :]),
        }
        for index, previous in enumerate(unit.previous_units):
            prior, prior_names = describe_controller(previous)
            priors.append(prior)
            state.update(_prefix(prior["state"], f"previous_units.{index}."))
            for key, value in prior_names.items():
                names[f"unit.previous_units.{index}.controller.{key}"] = f"previous_units.{index}.{value}"
        networks = {}
        for name in ("headout_wq", "headout_wk", "headin_wq", "headin_wk", "headin_wv", "internal_policy"):
            child = getattr(unit, name)
            state.update(_prefix(child.state_dict(), name + "."))
            networks[name] = _layout(child)
            names.update({f"unit.{name}.{key}": f"{name}.{key}" for key, _ in child.named_parameters()})
        networks["attention_temperature"] = float(unit.att_temp)
        node = _node(
            "CompoNetCore",
            state,
            priors=priors,
            hidden_dim=unit.headout_wq.out_features,
            equations=tb.EQUATIONS["componet"],
            networks=networks,
        )
        return node, names
    if isinstance(module, MoEActor):
        blocks = [tb.plain(expert._bundle_node) for expert in module.kaes]
        bank_state, slices, names, offset = {}, [], {}, 0
        for index, (block, expert) in enumerate(zip(blocks, module.kaes)):
            slices.append([offset, offset + block["mode_count"]])
            offset += block["mode_count"]
            bank_state.update(_prefix(block["state"], f"blocks.{index}."))
            for key, value in expert._bundle_names.items():
                names[f"kaes.{index}.{key}"] = f"kae.blocks.{index}.{value}"
        bank = _node(
            "StackedPortableKAE",
            bank_state,
            blocks=blocks,
            origins=[expert._bundle_origin for expert in module.kaes],
            mode_slices=slices,
        )
        state = _prefix(bank_state, "kae.")
        for name in ("mlp_network", "expert_weight_networks", "kae_router_network", "gating_network"):
            child = getattr(module, name)
            state.update(_prefix(child.state_dict(), f"head.{name}."))
            names.update({f"{name}.{key}": f"head.{name}.{key}" for key, _ in child.named_parameters()})
        networks = {
            "mlp": _layout(module.mlp_network),
            "mode_weight_networks": [_layout(network) for network in module.expert_weight_networks],
            "router": _layout(module.kae_router_network),
            "gate": _layout(module.gating_network),
            "branch_mode_counts": [block["mode_count"] for block in blocks],
            "router_activation": "softmax",
            "mode_weight_normalization": "signed_l1_epsilon_1e-6",
            "gate_activation": "sigmoid",
        }
        node = _node(
            "ExpandingKaeMoECore",
            state,
            kae=bank,
            steps=1,
            pad_value=1.0,
            residual_scale=1.0,
            sensitivity_scale=0.0,
            step_dt=0.02,
            equations=tb.EQUATIONS["kae_moe"],
            networks=networks,
        )
        return node, names
    if isinstance(module, SquashedActor):
        configuration = {
            "hidden_dims": _hidden_dims(_layout(module.trunk)),
            "normalization_state": None,
            "normalizer_epsilon": 1e-8,
            "normalizer_clip_threshold": 5.0,
        }
        node = _node(
            "FameActor",
            module.state_dict(),
            configuration=configuration,
            equations=tb.EQUATIONS["fame"],
            networks={"trunk": _layout(module.trunk)},
        )
        return node, {key: key for key, _ in module.named_parameters()}
    raise ValueError(f"Cannot describe a policy network of type {type(module).__name__} for a transfer bundle.")


def completed_controller(bundle: Mapping, device="cpu") -> nn.Module:
    """The complete deterministic policy of a bundle, frozen: observation to action, without the action noise."""
    node, method = bundle["architecture"]["policy"], bundle["manifest"]["method"]
    if method in ("residual", "componet", "kae_moe"):
        kind, prefix = {
            "residual": ("FrozenBaseResidualCore", "residual_core."),
            "componet": ("CompoNetCore", "componet_core."),
            "kae_moe": ("ExpandingKaeMoECore", "moe_core."),
        }[method]
        fields = {
            key: value
            for key, value in node.items()
            if key not in ("kind", "state", "min_log_std", "policy_initialization")
        }
        node = {"kind": kind, "state": _strip(node["state"], prefix), **fields}
    elif method == "finetune" and bundle["agent_modules"].get("state_preprocessor") is not None:
        checkpoint = finetune_checkpoint(bundle)
        node = ppo_controller_node(checkpoint)
    controller = controller_from_node(node, device)
    if method == "fame":
        controller = SquashedMean(controller)
    return freeze(controller)


# -- parents: what a method starts a new shape from ----------------------------------------------------------


def seed_from_square(
    method: Method,
    *,
    checkpoint: str | None = None,
    kae: str | None = None,
    policy: str | None = None,
    meta_policy: str | None = None,
) -> dict:
    """Give ``method`` the state after a Square stage, from the files of that stage named in :data:`SQUARE_FILES`.

    The methods in :data:`SHARED_SQUARE_METHODS` train the same plain PPO policy in their own first stage, so
    one Square trained with ``--method scratch`` serves all of them: ``checkpoint`` is its agent checkpoint
    (``checkpoints/best_agent.pt``) and ``kae`` the KAE of ``scripts/skrl/fit_kae.py`` (``kae/kae.pt``). FAME
    takes ``policy.pt`` and ``meta_policy.pt`` of its own Square stage. Returns the files that were read.
    """
    given = {"checkpoint": checkpoint, "kae": kae, "policy": policy, "meta_policy": meta_policy}
    given = {name: os.path.abspath(path) for name, path in given.items() if path}
    needed = SQUARE_FILES[method.name]
    if sorted(given) != sorted(needed):
        raise ValueError(
            f"Method '{method.name}' starts from the Square with {', '.join('--square_' + name for name in needed)}; "
            f"got {', '.join('--square_' + name for name in given) or 'none of them'}."
        )
    for name, path in given.items():
        if not os.path.isfile(path):
            raise FileNotFoundError(f"--square_{name}: no file {path}")
    if method.name == "finetune":
        method.checkpoint = given["checkpoint"]
    elif method.name == "kae_moe":
        origin = "simulation-square:" + hashlib.sha256(Path(given["kae"]).read_bytes()).hexdigest()
        method.experts = [kae_expert_from_node(kae_node(method.load_module(given["kae"])), origin, method.device)]
    elif method.name == "fame":
        # as Fame.resume after one stage
        method.agent.meta_actor.load_state_dict(method.load_module(given["meta_policy"]).actor.state_dict())
        method.agent._meta_trained = True
        method.agent.meta_buffer.clear()
        method.agent.meta_buffer.add(method._visited_states(method.load_module(given["policy"]).actor, "square"))
    else:
        loaded = torch.load(given["checkpoint"], map_location="cpu", weights_only=False)
        controller = freeze(controller_from_node(ppo_controller_node(loaded), method.device))
        if method.name == "residual":
            method.base = controller
        else:
            method.units = [PriorUnit(controller)]
    return given


def check_parent_bundle(bundle: Mapping, method_name: str, task: str):
    """Raise unless ``bundle`` is the finalized real result of ``method_name`` that the shape ``task`` starts from."""
    manifest = bundle["manifest"]
    if manifest["method"] != method_name:
        raise ValueError(f"The parent bundle belongs to method '{manifest['method']}', not '{method_name}'.")
    if manifest["phase"] not in ("real", "kinematic"):
        raise ValueError("The parent bundle must be a finished real stage, not a simulation warm-up.")
    next_stage = manifest.get("next_stage")
    if not isinstance(next_stage, Mapping):
        raise ValueError("The parent bundle is not finalized: run finalize_stage.py of Drone_Test_TRO on the real run.")
    if next_stage.get("strategy") != NEXT_STAGE_STRATEGY[method_name]:
        raise ValueError("The parent bundle was finalized for a different method.")
    if next_stage.get("shape") not in (None, task):
        raise ValueError(f"The parent bundle was finalized for shape '{next_stage.get('shape')}', not '{task}'.")
    if manifest["phase"] == "kinematic":
        print("[WARNING] The parent bundle comes from a dry run, not from a real flight.")


def finetune_checkpoint(bundle: Mapping) -> dict:
    """Agent checkpoint of this repository (policy, value, optimizer, normalizers) from a PPO fine-tuning bundle."""
    modules = bundle["agent_modules"]
    names = {"log_std_parameter": "log_std_parameter"}
    for native, canonical in (
        ("actor.0.", "net_container.0."),
        ("actor.2.", "net_container.2."),
        ("actor.4.", "policy_layer.0."),
    ):
        names.update({native + suffix: canonical + suffix for suffix in ("weight", "bias")})
    checkpoint = {
        "policy": {native: modules["policy"][canonical] for native, canonical in names.items()},
        "value": dict(modules["value"]),
    }
    # the optimizer of this repository holds the policy parameters, then the value parameters
    native_order = ["policy." + names[key] for key in checkpoint["policy"]] + [
        "value." + key for key in checkpoint["value"]
    ]
    saved = modules["optimizer"]
    canonical_order = bundle["agent_configuration"]["optimizer_parameter_names"]["optimizer"][0]
    states = {
        name: saved["state"].get(index) for name, index in zip(canonical_order, saved["param_groups"][0]["params"])
    }
    if set(native_order) != set(canonical_order):
        raise ValueError("The optimizer of the parent bundle does not hold the parameters of the waypoint PPO policy.")
    group = {key: value for key, value in saved["param_groups"][0].items() if key != "params"}
    group["params"] = list(range(len(native_order)))
    checkpoint["optimizer"] = {
        "state": {index: states[name] for index, name in enumerate(native_order) if states[name] is not None},
        "param_groups": [group],
    }
    for name in ("state_preprocessor", "value_preprocessor"):
        if modules.get(name) is not None:
            checkpoint[name] = dict(modules[name])
    return checkpoint


def seed_from_bundle(method: Method, bundle: Mapping, task: str, work_dir: str):
    """Give ``method`` the state after the finished real stage in ``bundle``, to train ``task`` next."""
    check_parent_bundle(bundle, method.name, task)
    next_stage = bundle["manifest"]["next_stage"]
    if method.name == "finetune":
        os.makedirs(work_dir, exist_ok=True)
        method.checkpoint = os.path.join(work_dir, "parent_agent.pt")
        torch.save(finetune_checkpoint(bundle), method.checkpoint)
    elif method.name == "residual":
        method.base = completed_controller(bundle, method.device)
    elif method.name == "componet":
        priors = [controller_from_node(prior, method.device) for prior in bundle["architecture"]["policy"]["priors"]]
        method.units = [PriorUnit(freeze(prior)) for prior in priors + [completed_controller(bundle, method.device)]]
    elif method.name == "kae_moe":
        bank = bundle["architecture"]["policy"]["kae"]
        nodes, origins = list(bank["blocks"]), list(bank["origins"])
        if "appended_kae" not in next_stage:
            raise ValueError("The parent bundle has no KAE of the real-adapted policy; finalize it with the KAE files.")
        nodes.append(next_stage["appended_kae"])
        record = next_stage.get("appended_kae_record") or {}
        origins.append("completed-real:" + record.get("sha256", tb.digest(next_stage["appended_kae"])))
        if len(set(origins)) != len(origins):
            raise ValueError("The KAE bank of the parent bundle contains the same KAE twice.")
        method.experts = [kae_expert_from_node(node, origin, method.device) for node, origin in zip(nodes, origins)]
    else:  # fame
        state = next_stage.get("consolidated_fame_state")
        if state is None:
            raise ValueError("The parent bundle has no consolidated FAME state; finalize the real stage first.")
        agent = method.agent
        configuration = state["actor_configuration"]
        if configuration.get("normalization_state") is not None:
            raise ValueError("A FAME actor with an input normalizer is not supported: FAME here uses raw observations.")
        if list(configuration["hidden_dims"]) != list(agent.hidden_dims):
            raise ValueError(
                f"The parent FAME actor has hidden sizes {list(configuration['hidden_dims'])}, the configuration "
                f"(agent.methods.fame.hidden_dims) {list(agent.hidden_dims)}."
            )
        agent.meta_actor.load_state_dict(state["meta_actor"])
        agent._meta_trained = bool(state["meta_trained"])
        saved = state["meta_buffer"]
        agent.meta_buffer = StateBuffer(saved["data"].shape[0], agent.obs_dim, agent.device)
        agent.meta_buffer.data.copy_(saved["data"].to(agent.device))
        agent.meta_buffer.index, agent.meta_buffer.size = int(saved["index"]), int(saved["size"])


# -- export of a warmed learner -----------------------------------------------------------------------------


class _Temperature:
    """Lets skrl checkpoints save and restore FAME's entropy temperature, which is a bare tensor."""

    def __init__(self, agent):
        self.agent = agent

    def state_dict(self):
        return {"log_alpha": self.agent.log_alpha.detach().clone()}

    def load_state_dict(self, state):
        with torch.no_grad():
            self.agent.log_alpha.copy_(state["log_alpha"])


def track_fame_training_state(agent):
    """Add FAME's optimizers and temperature to its checkpoints.

    FAME checkpoints hold only the actors and the critic, which is all a finished stage needs. A warm-up hands
    over a learner that keeps training, so its best checkpoint must also hold the matching optimizer states.
    Call this after ``start_stage``, which creates the fast learner's optimizers.
    """
    agent.checkpoint_modules["fast_actor_optimizer"] = agent.fast_actor_opt
    agent.checkpoint_modules["fast_critic_optimizer"] = agent.fast_critic_opt
    agent.checkpoint_modules["temperature"] = _Temperature(agent)
    agent.checkpoint_modules["temperature_optimizer"] = agent.log_alpha_opt


def _remap_adam(optimizer: torch.optim.Optimizer, native_names: Mapping, canonical: list[str]) -> dict:
    """State of an Adam optimizer with its parameters in the order ``canonical`` (names of ``Drone_Test_TRO``).

    ``native_names`` maps ``id(parameter)`` to the canonical name. Frozen parameters have no optimizer state;
    a trainable parameter without a canonical name is an error.
    """
    if not isinstance(optimizer, torch.optim.Adam) or len(optimizer.param_groups) != 1:
        raise ValueError("A transfer bundle holds Adam optimizers with one parameter group.")
    saved = optimizer.state_dict()
    states = {}
    for index, parameter in zip(saved["param_groups"][0]["params"], optimizer.param_groups[0]["params"]):
        name = native_names.get(id(parameter))
        if name is None or name not in canonical:
            if parameter.requires_grad:
                raise ValueError("A trainable parameter has no counterpart in the transfer bundle.")
            continue
        if index in saved["state"]:
            states[name] = saved["state"][index]
    group = {key: value for key, value in saved["param_groups"][0].items() if key != "params"}
    group["params"] = list(range(len(canonical)))
    return tb.plain(
        {
            "state": {index: states[name] for index, name in enumerate(canonical) if name in states},
            "param_groups": [group],
        }
    )


def _scheduler_records(agent) -> tuple[dict | None, dict | None]:
    """Kind and settings of the learning-rate schedule, and its state, for the bundle."""
    scheduler = getattr(agent, "scheduler", None)
    if scheduler is None:
        return None, None
    kind = type(scheduler).__name__
    if kind == "KLAdaptiveLR":
        record = {
            "kind": kind,
            "kl_threshold": float(scheduler.kl_threshold),
            "min_lr": float(scheduler.min_lr),
            "max_lr": float(scheduler.max_lr),
            "kl_factor": float(scheduler._kl_factor),
            "lr_factor": float(scheduler._lr_factor),
        }
    elif kind == "ExponentialLR":
        record = {"kind": kind, "gamma": float(scheduler.gamma)}
    else:
        raise ValueError(f"A transfer bundle cannot hold a {kind} learning-rate schedule.")
    state = {}
    for key, value in scheduler.state_dict().items():
        try:
            state[key] = tb.plain(value)
        except ValueError:
            continue  # bookkeeping of torch that is not a number, e.g. wrapped functions
    # the current rate is in the optimizer; the checkpoint that was loaded may be older than the schedule object
    state["_last_lr"] = [float(group["lr"]) for group in agent.optimizer.param_groups]
    return record, state


def describe_learner(method_name: str, agent) -> tuple[dict, dict]:
    """Description of the policy that is being trained (with its action noise) and its parameter-name map."""
    if method_name == "fame":
        return describe_controller(agent.fast_actor)
    policy = agent.policy
    min_log_std = float(policy._min_log_std)
    if method_name == "finetune":
        actor = policy.actor
        state = {"log_std_parameter": policy.log_std_parameter}
        names = {"log_std_parameter": "log_std_parameter"}
        for native, canonical in (("0.", "net_container.0."), ("2.", "net_container.2."), ("4.", "policy_layer.0.")):
            for suffix in ("weight", "bias"):
                state[canonical + suffix] = actor.state_dict()[native + suffix]
                names["actor." + native + suffix] = canonical + suffix
        node = _node(
            "CrazySimWaypointPolicy",
            state,
            min_log_std=min_log_std,
            equations=tb.EQUATIONS["finetune"],
            networks={"hidden": _layout(actor[:4]), "output": _layout(actor[4:]), "output_transform": "as_declared"},
        )
        return node, names
    core, core_names = describe_controller(policy.actor)
    kind, core_kind, prefix = {
        "residual": ("ResidualPolicy", "FrozenBaseResidualCore", "residual_core."),
        "componet": ("CompoNetPolicy", "CompoNetCore", "componet_core."),
        "kae_moe": ("ExpandingKaeMoEPolicy", "ExpandingKaeMoECore", "moe_core."),
    }[method_name]
    if core["kind"] != core_kind:
        raise ValueError(f"The policy of method '{method_name}' has no earlier policy to build on; give it a parent.")
    state = {
        "log_std_parameter": policy.log_std_parameter,
        **_prefix(core["state"], prefix),
        **_prefix(core["state"], "policy_layer."),
    }
    fields = {key: value for key, value in core.items() if key not in ("kind", "state", "networks", "equations")}
    if method_name == "residual":
        fields["policy_initialization"] = "legacy"
    node = _node(
        kind, state, min_log_std=min_log_std, equations=tb.EQUATIONS[method_name], networks=core["networks"], **fields
    )
    names = {
        "log_std_parameter": "log_std_parameter",
        **{"actor." + key: prefix + value for key, value in core_names.items()},
    }
    return node, names


def _fame_records(agent, policy: Mapping, contract: str) -> tuple[dict, dict, dict, dict]:
    """Value description, saved modules, settings and optimizer parameter names of a FAME agent."""
    value = _node(
        "FameValue",
        _prefix(agent.fast_critic.state_dict(), "net."),
        hidden_dims=list(agent.hidden_dims),
        networks={"value": _layout(agent.fast_critic)},
    )
    policy_names, value_names = _parameter_names(policy["state"], "policy"), _parameter_names(value["state"], "value")
    actor_ids = {id(parameter): "policy." + name for name, parameter in agent.fast_actor.named_parameters()}
    critic_ids = {id(parameter): "value.net." + name for name, parameter in agent.fast_critic.named_parameters()}
    buffer = lambda b: {"data": tb.plain(b.data), "index": int(b.index), "size": int(b.size)}  # noqa: E731
    modules = {
        "policy": policy["state"],
        "value": value["state"],
        "optimizer": _remap_adam(agent.fast_actor_opt, actor_ids, policy_names),
        "critic_optimizer": _remap_adam(agent.fast_critic_opt, critic_ids, value_names),
        "temperature": {"log_alpha": tb.plain(agent.log_alpha)},
        "temperature_optimizer": _remap_adam(
            agent.log_alpha_opt, {id(agent.log_alpha): "temperature.log_alpha"}, ["temperature.log_alpha"]
        ),
        "meta_actor": describe_controller(agent.meta_actor)[0]["state"],
        "fame_buffers": {
            "version": 2,
            "core_equations_version": tb.EQUATIONS["fame"],
            "fast2meta": buffer(agent.fast2meta),
            "meta": buffer(agent.meta_buffer),
            "meta_trained": bool(agent._meta_trained),
            "games": [],
            "flag_reg": [],
            "actor_configuration": policy["configuration"],
            "policy_initialization": None,
        },
    }
    settings = {
        "rollouts": int(agent.cfg["rollout_steps"]),
        "mixed_precision": False,
        "fame_size_fast2meta": int(agent.fast2meta.capacity),
        "fame_size_meta": int(agent.meta_buffer.capacity),
        "fame_observation_contract": contract,
        "fame_core": {key: tb.plain(item) for key, item in agent.cfg.items() if key != "experiment"},
        "rewards_shaper_scale": agent.cfg["rewards_shaper_scale"],
    }
    names = {
        "optimizer": [policy_names],
        "critic_optimizer": [value_names],
        "temperature_optimizer": [["temperature.log_alpha"]],
    }
    return value, modules, settings, names


def _ppo_records(agent, policy: Mapping, names: Mapping) -> tuple[dict, dict, dict, dict, dict]:
    """Value description, saved modules, settings, optimizer parameter names and normalizers of a PPO agent."""
    value = _node(
        "BoundedValue",
        agent.value.state_dict(),
        hidden_dims=_hidden_dims(_layout(agent.value.net)),
        networks={"value": _layout(agent.value.net)},
    )
    canonical = _parameter_names(policy["state"], "policy") + _parameter_names(value["state"], "value")
    ids = {id(p): "policy." + names[name] for name, p in agent.policy.named_parameters() if name in names}
    ids.update({id(p): "value." + name for name, p in agent.value.named_parameters()})
    modules = {
        "policy": policy["state"],
        "value": value["state"],
        "optimizer": _remap_adam(agent.optimizer, ids, canonical),
    }
    preprocessors = {}
    for key in ("state_preprocessor", "value_preprocessor"):
        preprocessor = getattr(agent, "_" + key)
        if hasattr(preprocessor, "state_dict"):
            preprocessors[key] = {
                "kind": "SingleDroneRunningStandardScaler",
                "size": preprocessor.running_mean.numel(),
                "epsilon": float(preprocessor.epsilon),
                "clip_threshold": float(preprocessor.clip_threshold),
            }
            modules[key] = tb.plain(preprocessor.state_dict())
        else:
            preprocessors[key] = None
    skipped = (
        "experiment", "state_preprocessor", "state_preprocessor_kwargs", "value_preprocessor",
        "value_preprocessor_kwargs", "learning_rate_scheduler", "learning_rate_scheduler_kwargs", "rewards_shaper",
    )  # fmt: skip
    settings = {}
    for key, item in agent.cfg.items():
        if key in skipped:
            continue
        try:
            settings[key] = tb.plain(item)
        except ValueError:
            continue  # classes and functions that skrl keeps in the configuration
    return value, modules, settings, {"optimizer": [canonical]}, preprocessors


def environment_contract(env_cfg, task: str) -> dict:
    """Task settings of the waypoint environment in the terms of ``Drone_Test_TRO``, for the bundle manifest.

    ``Drone_Test_TRO`` compares these with the settings of the real run that imports the bundle.
    """
    tasks = env_cfg.tasks
    square, triangle = tasks["square"].vertices, tasks["triangle"].vertices
    side = lambda a, b: round(float(((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5), 6)  # noqa: E731
    active = tasks[task]
    if task == "circle":
        length = 2.0 * torch.pi * active.radius
    elif task == "figure8":
        # midpoint rule over the curve length, as figure8_path_length of Drone_Test_TRO
        a, b, samples = active.size[0], active.size[1], 4096
        angles = (torch.arange(samples, dtype=torch.float64) + 0.5) * (2.0 * torch.pi / samples)
        length = float(
            torch.hypot(a * torch.cos(angles), 2.0 * b * torch.cos(2.0 * angles)).sum() * 2.0 * torch.pi / samples
        )
    else:
        corners = active.vertices
        length = sum(side(corners[i], corners[(i + 1) % len(corners)]) for i in range(len(corners)))
    contract = {
        "control_period_s": round(float(env_cfg.sim.dt) * int(env_cfg.decimation), 6),
        "target_height": float(active.center[2]),
        "trajectory_scale": 1.0,
        "trajectory_speed": round(float(length) / float(active.period), 6),
        "trajectory_laps": int(env_cfg.trajectory_laps),
        "square_side_length": side(square[0], square[1]),
        "circle_radius": float(tasks["circle"].radius),
        "triangle_side_length": side(triangle[0], triangle[1]),
        "figure8_amp_x": float(tasks["figure8"].size[0]),
        "figure8_amp_y": float(tasks["figure8"].size[1]),
        "figure8_random_height": False,
        "observation_mode": env_cfg.observation_mode,
        "trajectory_setpoint_count": int(env_cfg.setpoint_count),
        "waypoint_offset_limit": [float(x) for x in env_cfg.waypoint_offset_limit],
        "observation_position_error_scale": [float(x) for x in env_cfg.observation_position_scale],
        "observation_velocity_scale": float(env_cfg.observation_velocity_scale),
        "action_clip_min": -1.0,
        "action_clip_max": 1.0,
        "command_constraint": env_cfg.command_constraint,
        "command_rate_limit": [float(x) for x in env_cfg.command_rate_limit],
        "command_proximity_radius": [float(x) for x in env_cfg.command_proximity_radius],
        "position_cmd_xy_min": -float(env_cfg.command_xy_limit),
        "position_cmd_xy_max": float(env_cfg.command_xy_limit),
        "position_cmd_z_min": float(env_cfg.command_z_range[0]),
        "position_cmd_z_max": float(env_cfg.command_z_range[1]),
        "shape": task,
        "command_semantics": "world_position_setpoint_relative_waypoint_v1",
        "yaw_radians": 0.0,
        "reference_geometry_version": "fixed_center_five_shapes_v1",
        "shape_rotation_y_degrees": 0.0,
    }
    return contract


def observation_contract(environment: Mapping) -> str:
    """``observation_contract`` of ``examples/trajectory_setpoints/observation.py`` in ``Drone_Test_TRO``."""
    limits = ",".join(f"{value:g}" for value in environment["waypoint_offset_limit"])
    count = environment["trajectory_setpoint_count"]
    return f"trajectory_setpoints_relative_waypoint_v1:setpoints={count}:offset_limit={limits}"


def _source_revision() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent, text=True, capture_output=True, check=False
    )
    return result.stdout.strip() or "unknown"


def export_bundle(
    method_name: str,
    agent,
    path: str,
    *,
    task: str,
    environment: Mapping,
    num_envs: int,
    budget_iterations: int,
    selected_iteration: int,
    parent: Mapping | None = None,
    parent_square: Mapping | None = None,
) -> Path:
    """Write the learner in ``agent`` as a simulation ``transfer_bundle.pt`` for the same shape on the real drone.

    Args:
        budget_iterations: Training iterations of the warm-up.
        selected_iteration: Iteration of the checkpoint that is in ``agent`` (at most ``budget_iterations``).
        parent: The finalized real bundle this warm-up started from, if any.
        parent_square: Files of the simulated Square this warm-up started from (:func:`seed_from_square`), if any.
    """
    if task not in tb.SHAPES:
        raise ValueError(f"Drone_Test_TRO has no shape '{task}'.")
    contract = observation_contract(environment)
    policy, names = describe_learner(method_name, agent)
    if method_name == "fame":
        value, modules, settings, optimizer_names = _fame_records(agent, policy, contract)
        preprocessors = {"state_preprocessor": None, "value_preprocessor": None}
        scheduler_record, scheduler_state, scaler_state = None, None, {}
        learning_rate = float(agent.fast_actor_opt.param_groups[0]["lr"])
    else:
        value, modules, settings, optimizer_names, preprocessors = _ppo_records(agent, policy, names)
        scheduler_record, scheduler_state = _scheduler_records(agent)
        scaler = getattr(agent, "scaler", None)
        scaler_state = None if scaler is None else tb.plain(scaler.state_dict())
        learning_rate = float(agent.optimizer.param_groups[0]["lr"])
    rollouts = int(settings["rollouts"])
    per_iteration = rollouts * int(num_envs)
    counters = {"accepted_transitions": selected_iteration * per_iteration, "complete_updates": int(selected_iteration)}
    cumulative, parent_record = dict(counters), None
    if parent is not None:
        manifest = parent["manifest"]
        parent_record = {
            "artifact_id": manifest["artifact_id"],
            "method": method_name,
            "shape": manifest["shape"],
            "phase": manifest["phase"],
            **tb.plain(parent.get("_source", {})),
        }
        for key, amount in manifest["cumulative_counters"].items():
            if type(amount) is int:
                cumulative[key] = cumulative.get(key, 0) + amount
    manifest = {
        "version": tb.VERSION,
        "artifact_type": tb.ARTIFACT_TYPE,
        "method": method_name,
        "equations": tb.EQUATIONS[method_name],
        "source_revision": _source_revision(),
        "source_dirty_diff_sha256": None,
        "source_adapter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "shape": task,
        "phase": "simulation",
        "stage_identity": f"{method_name}:{task}:simulation:{Path(path).parent.name}",
        "parent": parent_record,
        "simulation_parent": None if parent_square is None else {"shape": "square", "files": dict(parent_square)},
        "dependencies": tb.dependency_records(policy),
        "observation_contract": contract,
        "observation_dim": OBS_DIM,
        "action_dim": ACTION_DIM,
        "observation_specification": tb.OBSERVATION_SPECIFICATION,
        "action_contract": {
            "bounds": [-1.0, 1.0],
            "decoder": "position_plus_scaled_action",
            "offset_limit_m": environment["waypoint_offset_limit"],
            "yaw_radians": 0.0,
        },
        "distribution": tb.distribution_contract(method_name, policy),
        "environment": tb.plain(environment),
        "phase_counters": counters,
        "counters": counters,
        "cumulative_counters": cumulative,
        "budget": {
            "max_episodes": None,
            "accepted_transitions": budget_iterations * per_iteration,
            "complete_updates": int(budget_iterations),
            "allocated_iterations": int(budget_iterations),
            "num_envs": int(num_envs),
            "rollout_steps": rollouts,
        },
        "checkpoint_selection": {
            "rule": "best_training_return_within_budget",
            "actual_update": int(selected_iteration),
            "actual_iteration": int(selected_iteration),
        },
        "learning_rate_at_selection": learning_rate,
        "resets": ["on_policy_rollout", "backend_environment", "backend_rng", "phase_counters"],
        "preserved_method_memory": ["fast2meta", "meta"] if method_name == "fame" else [],
        "next_stage": None,
    }
    bundle = {
        "manifest": manifest,
        "architecture": {"policy": policy, "value": value},
        "agent_modules": modules,
        "agent_configuration": {
            "settings": settings,
            "preprocessors": preprocessors,
            "scheduler": scheduler_record,
            "optimizer_types": {key: "Adam" for key in optimizer_names},
            "optimizer_parameter_names": optimizer_names,
        },
        "scheduler": scheduler_state,
        "grad_scaler": scaler_state,
    }
    return tb.write_bundle(bundle, path)


def finish_warm_stage(
    method: Method,
    agent,
    stage_dir: str,
    task: str,
    env_cfg,
    *,
    budget_iterations: int,
    parent: Mapping | None = None,
    parent_square: Mapping | None = None,
) -> tuple[nn.Module, dict]:
    """End a warm-up stage: select the best checkpoint and export it. Used instead of ``Method.end_stage``.

    Returns the deterministic policy of the selected checkpoint and the entries for the stage in
    ``session.json``. The method keeps no state for a later stage: the next shape starts from the finalized
    real result (:func:`seed_from_bundle`).
    """
    if getattr(method, "_gate_hook", None) is not None:  # KAE-MoE logging hook; it would be saved with the policy
        method._gate_hook.remove()
        method._gate_hook = None
    rollouts = int(method.agent_cfg["agent"]["rollouts"])
    selected_path = method.load_best_checkpoint(agent, stage_dir)
    is_best = os.path.basename(selected_path) == "best_agent.pt"
    selected_iteration = int(method.stage_info["best_timestep"]) // rollouts if is_best else int(budget_iterations)
    if method.name == "fame":
        if selected_iteration < budget_iterations:
            # as Fame.end_stage: the states recorded during training were visited by later actors
            agent.fast2meta.clear()
            agent.fast2meta.add(method._visited_states(agent.fast_actor, task))
        policy = SquashedMean(copy.deepcopy(agent.fast_actor))
    else:
        policy = method.ppo_policy(agent)
    bundle_path = export_bundle(
        method.name,
        agent,
        os.path.join(stage_dir, tb.BUNDLE_FILE),
        task=task,
        environment=environment_contract(env_cfg, task),
        num_envs=method.env.num_envs,
        budget_iterations=budget_iterations,
        selected_iteration=selected_iteration,
        parent=parent,
        parent_square=parent_square,
    )
    optimizer = agent.fast_actor_opt if method.name == "fame" else agent.optimizer
    info = {
        **method.stage_info,
        "warm_start": True,
        "budget_iterations": int(budget_iterations),
        "selected_iteration": selected_iteration,
        "learning_rate_at_selection": float(optimizer.param_groups[0]["lr"]),
        "transfer_bundle": str(bundle_path),
    }
    print(
        f"[INFO] Warm-up of '{task}' ({method.name}): checkpoint of iteration {selected_iteration} of "
        f"{budget_iterations}, learning rate {info['learning_rate_at_selection']:.3g}"
    )
    print(f"[INFO] Wrote {bundle_path}")
    return policy.eval(), info
