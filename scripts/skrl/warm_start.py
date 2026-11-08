"""Warm one simulation shape, then export its current learner for real continuation.

This additive launcher never calls a method's end_stage. Run --help without
Isaac Lab; actual physics execution requires the normal CrazySim environment.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from contextlib import contextmanager
import hashlib
from functools import wraps
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

METHODS = ("kae_moe", "finetune", "residual", "componet", "fame")
SHAPES = ("square", "circle", "triangle", "figure8", "skewed_figure8")


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--shape", choices=SHAPES, required=True)
    entry = parser.add_mutually_exclusive_group(required=True)
    entry.add_argument("--parent-bundle", type=Path,
                       help="Finalized completed real stage for the next shape.")
    entry.add_argument("--prepare-square", action="store_true",
                       help="Explicit extra preparation from shared Square source inputs.")
    parser.add_argument("--square-policy", type=Path)
    parser.add_argument("--square-kae", type=Path)
    parser.add_argument("--iterations", type=positive_int, required=True,
                        help="Complete native training updates; choose with the KAE-MoE pilot.")
    parser.add_argument("--num-envs", type=positive_int, required=True)
    parser.add_argument("--rollout-steps", type=positive_int, default=96)
    parser.add_argument("--checkpoint-selection", choices=("exact",), required=True,
                        help="Export the exact final update, without best-policy replacement.")
    parser.add_argument("--new-shape-noise", choices=("reset", "preserve", "original"), required=True,
                        help="Explicit new-shape exploration decision; FAME requires original task detection.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--evaluate", action="store_true",
                        help="Run the existing deterministic evaluator after exporting (outside training data).")
    parser.add_argument("--allow-kinematic-parent", action="store_true",
                        help="Accept a software-only parent for integration validation; result stays software-only.")
    parser.add_argument("--validate-only", action="store_true",
                        help="Validate inputs and lineage without starting Isaac Sim or training.")
    parser.add_argument("--detection-steps", type=positive_int,
                        help="Override FAME's original per-candidate detection steps; reported separately.")
    for flag in ("square-side-length", "circle-radius", "triangle-side-length", "figure8-amp-x",
                 "figure8-amp-y", "target-height", "trajectory-speed", "trajectory-scale"):
        parser.add_argument("--" + flag, type=float)
    parser.add_argument("--trajectory-laps", type=positive_int)
    return parser


def validate_arguments(args, parser):
    if args.prepare_square:
        if args.shape != "square":
            parser.error("--prepare-square applies only to --shape square.")
        if args.square_policy is None:
            parser.error("Square preparation requires --square-policy, a complete shared source policy.")
        if args.method == "kae_moe" and args.square_kae is None:
            parser.error("Square KAE-MoE preparation also requires --square-kae.")
    elif args.square_policy is not None or args.square_kae is not None:
        parser.error("Square source inputs apply only to --prepare-square.")
    if args.method == "fame" and args.new_shape_noise != "original":
        parser.error("FAME uses --new-shape-noise original: retain the fresh/meta detection winner's Gaussian.")
    if args.method != "fame" and args.new_shape_noise == "original":
        parser.error("Select --new-shape-noise reset or preserve for this method.")
    for path in (args.parent_bundle, args.square_policy, args.square_kae):
        if path is not None and not path.expanduser().is_file():
            parser.error(f"Input file not found: {path}")
    if args.num_envs * args.rollout_steps < 2:
        parser.error("A native update needs at least two transitions for finite advantage statistics.")
    for key in ("square_side_length", "circle_radius", "triangle_side_length", "figure8_amp_x", "figure8_amp_y",
                "target_height", "trajectory_speed", "trajectory_scale"):
        value = getattr(args, key)
        if value is not None and (not math.isfinite(value) or value <= 0):
            parser.error(f"--{key.replace('_', '-')} must be finite and positive.")
    if args.method != "fame" and args.detection_steps is not None:
        parser.error("--detection-steps applies only to FAME.")


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w") as stream:
            json.dump(value, stream, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def configuration_snapshot(value):
    """Describe resolved constructors without serializing executable objects."""
    if isinstance(value, dict):
        return {key: configuration_snapshot(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [configuration_snapshot(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if callable(value):
        return f"{value.__module__}.{value.__qualname__}"
    if value is None or type(value) in (str, bool, int, float):
        return value
    raise TypeError(f"Cannot record resolved configuration value {type(value).__name__}.")


def repository_revision(root):
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()


def _source_records(args):
    return [{"path": str(path.expanduser().resolve()),
             "sha256": hashlib.sha256(path.expanduser().read_bytes()).hexdigest()}
            for path in (args.square_policy, args.square_kae) if path is not None]


@contextmanager
def count_training_interactions(agent):
    """Observe original record/update calls without changing either algorithm."""
    counts = {"accepted_transitions": 0, "complete_updates": 0}
    record, update = agent.record_transition, agent._update

    def counted_record(*values, **kwargs):
        result = record(*values, **kwargs)
        states = kwargs.get("states", values[0] if values else None)
        counts["accepted_transitions"] += int(states.shape[0])
        return result

    def counted_update(*values, **kwargs):
        result = update(*values, **kwargs)
        counts["complete_updates"] += 1
        return result

    agent.record_transition, agent._update = counted_record, counted_update
    try:
        yield counts
    finally:
        agent.record_transition, agent._update = record, update


@contextmanager
def count_optimizer_steps(agent):
    """Report auxiliary actor/critic/temperature work independently of updates."""
    candidates = (("ppo", "optimizer"), ("actor", "fast_actor_opt"),
                  ("critic", "fast_critic_opt"), ("temperature", "log_alpha_opt"))
    counts, originals, seen = {}, [], set()
    for name, attribute in candidates:
        optimizer = getattr(agent, attribute, None)
        if optimizer is None or id(optimizer) in seen:
            continue
        seen.add(id(optimizer))
        counts[name] = 0
        original = optimizer.step

        @wraps(original)
        def observed(*values, _original=original, _name=name, **kwargs):
            result = _original(*values, **kwargs)
            counts[_name] += 1
            return result

        originals.append((optimizer, original))
        optimizer.step = observed
    try:
        yield counts
    finally:
        for optimizer, original in originals:
            optimizer.step = original


def run_native_phase(method, parent, args, environment_contract, output_dir, *, trainer_class=None):
    """One original task switch and one capped trainer invocation; no end_stage."""
    from warm_start_stage import start_warm_stage, deterministic_current_policy
    from warm_start_io import export_warmed_learner

    if trainer_class is None:
        from skrl.trainers.torch import SequentialTrainer
        trainer_class = SequentialTrainer
    output_dir = Path(output_dir)
    started = time.monotonic()
    agent, initialization = start_warm_stage(
        method, parent, args.shape, str(output_dir), prepare_square=args.prepare_square,
        square_policy=args.square_policy, square_kae=args.square_kae,
        new_shape_noise=args.new_shape_noise,
    )
    try:
        trainer_cfg = {key: value for key, value in method.agent_cfg["trainer"].items() if key != "class"}
        trainer_cfg.update(timesteps=args.iterations * args.rollout_steps,
                           close_environment_at_exit=False, disable_progressbar=True)
        # SequentialTrainer initializes the agent, clearing its on-policy rollout.
        # Method buffers and compatible optimizer state survive that initialization.
        trainer = trainer_class(env=method.env, agents=agent, cfg=trainer_cfg)
        with count_training_interactions(agent) as counters, count_optimizer_steps(agent) as optimizer_steps:
            trainer.train()
        elapsed = time.monotonic() - started
        expected = {"accepted_transitions": args.iterations * args.rollout_steps * args.num_envs,
                    "complete_updates": args.iterations}
        if counters != expected:
            raise RuntimeError(f"Warm-up did not reach its exact complete-update budget: {counters}; expected {expected}.")
        artifact = export_warmed_learner(
            method, agent, output_dir / "transfer_bundle.pt", parent=parent,
            shape=args.shape, environment=environment_contract, iterations=args.iterations,
            counters=counters, checkpoint_selection="exact_cutoff",
            source_revision=repository_revision(Path(__file__).resolve().parents[2]),
            initialization=initialization,
        )
        import torch
        policy_path = output_dir / "policy.pt"
        torch.save(deterministic_current_policy(method, agent), policy_path)
        checkpoint_dir = output_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        agent.save(str(checkpoint_dir / "final_agent.pt"))
        record = {
            "method": args.method, "shape": args.shape, "phase": "simulation",
            "software_validation_only": bool(parent is not None and (
                parent["manifest"]["phase"] == "kinematic" or parent["manifest"].get("software_validation_only")
                or (parent["manifest"].get("next_stage") or {}).get("software_validation_only"))),
            "entry": "square_preparation" if args.prepare_square else "next_shape_from_completed_real",
            "source_inputs": _source_records(args), "initialization": initialization,
            "training": {"allocated_iterations": args.iterations, "actual_iteration": args.iterations,
                         "rollout_steps": args.rollout_steps, "num_envs": args.num_envs,
                         "accepted_transitions": counters["accepted_transitions"],
                         "elapsed_seconds": elapsed, "checkpoint_selection": "exact_cutoff",
                         "optimizer_steps": optimizer_steps},
            "end_stage_called": False,
            "evaluation_policy": {"path": str(policy_path.resolve()),
                                  "sha256": hashlib.sha256(policy_path.read_bytes()).hexdigest()},
            "bundle": {"path": str(artifact.resolve()),
                       "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest()},
        }
        atomic_json(output_dir / "warm_start_summary.json", record)
        return agent, artifact, record
    finally:
        writer = getattr(agent, "writer", None)
        if writer is not None:
            try:
                writer.flush()
            finally:
                writer.close()


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_arguments(args, parser)
    from warm_start_io import load_real_parent, validate_square_inputs

    parent = (None if args.parent_bundle is None else load_real_parent(
        args.parent_bundle, args.method, args.shape, allow_kinematic=args.allow_kinematic_parent,
    ))
    square_validation = (validate_square_inputs(args.square_policy, args.square_kae)
                         if args.prepare_square else None)
    if args.validate_only:
        print(json.dumps({"validated": "input_schema_and_lineage", "physics_started": False,
                          "method": args.method, "shape": args.shape,
                          "parent_identity": None if parent is None else parent["manifest"]["artifact_id"],
                          "source_inputs": _source_records(args), "square_contract": square_validation}, indent=2))
        return None
    try:
        from isaaclab.app import AppLauncher
    except ImportError as error:
        raise RuntimeError("Isaac Lab is required for physics warm-up. Use its Python environment; "
                           "--validate-only checks input lineage without starting physics.") from error
    application = AppLauncher(args)
    simulation_app = application.app
    env = None
    try:
        import gymnasium as gym
        import yaml
        from skrl.utils import set_seed
        from isaaclab_rl.skrl import SkrlVecEnvWrapper
        import crazysim  # noqa: F401 - unchanged environment registration after app launch
        from crazysim.methods import METHODS as native_methods
        from crazysim.tasks.direct.tracking.waypoint_env_cfg import WaypointTrackingEnvCfg
        from warm_start_config import build_warm_start_configuration
        from warm_start_stage import configure_for_parent

        root = Path(__file__).resolve().parents[2]
        agent_dir = root / "source/crazysim/crazysim/tasks/direct/tracking/agents"
        agent_cfg = yaml.safe_load((agent_dir / "skrl_ppo_waypoint_cfg.yaml").read_text())
        general_cfg = yaml.safe_load((agent_dir / "skrl_ppo_cfg.yaml").read_text())
        geometry = {key: getattr(args, key) for key in (
            "square_side_length", "circle_radius", "triangle_side_length", "figure8_amp_x", "figure8_amp_y",
            "target_height", "trajectory_speed", "trajectory_scale", "trajectory_laps") if getattr(args, key) is not None}
        env_cfg, agent_cfg, contract = build_warm_start_configuration(
            WaypointTrackingEnvCfg(), agent_cfg, args.method, args.shape,
            general_agent_cfg=general_cfg, parent_environment=None if parent is None else parent["manifest"]["environment"],
            num_envs=args.num_envs, rollout_steps=args.rollout_steps, iterations=args.iterations, geometry=geometry,
        )
        agent_cfg = configure_for_parent(agent_cfg, parent, args.method, square_policy=args.square_policy)
        if args.detection_steps is not None:
            agent_cfg["methods"]["fame"]["detection_steps"] = args.detection_steps
        agent_cfg["seed"] = env_cfg.seed = args.seed
        env_cfg.sim.device = args.device
        output_dir = (args.output_dir.expanduser().resolve() if args.output_dir is not None else
                      root / "logs/warm_start" / f"{datetime.now(timezone.utc):%Y-%m-%d_%H-%M-%S}_{args.method}_{args.shape}")
        output_dir.mkdir(parents=True, exist_ok=False)
        agent_cfg["agent"]["experiment"]["directory"] = str(output_dir)
        agent_cfg["agent"]["experiment"]["checkpoint_interval"] = 0
        env_cfg.log_dir = str(output_dir)
        set_seed(args.seed)
        env = SkrlVecEnvWrapper(gym.make("CrazySim-TrackingWaypoint-Direct-v0", cfg=env_cfg), ml_framework="torch")
        method = native_methods[args.method](env, agent_cfg, str(output_dir))
        atomic_json(output_dir / "warm_start_config.json", {"arguments": {key: str(value) if isinstance(value, Path) else value
                    for key, value in vars(args).items()}, "environment_contract": contract,
                    "source_revision": repository_revision(root),
                    "resolved_agent_configuration": configuration_snapshot(agent_cfg),
                    "simulation_settings": {key: getattr(env_cfg, key, None) for key in (
                        "random_start_phase", "init_pos_noise", "max_position_error", "motor_time_constant",
                        "firmware_xy_velocity_max", "firmware_z_velocity_max")},
                    "physics_dt": env_cfg.sim.dt, "decimation": env_cfg.decimation})
        agent, artifact, record = run_native_phase(method, parent, args, contract, output_dir)
        if args.evaluate:
            from crazysim.evaluation import evaluate_tasks
            from warm_start_stage import deterministic_current_policy
            metrics = evaluate_tasks(env, deterministic_current_policy(method, agent), [args.shape])
            atomic_json(output_dir / "evaluation.json", {
                "metrics": metrics, "position_error_metric": "mean_distance_m_not_rmse",
                "training_transitions": 0, "normalization_updates": 0,
            })
        print(f"Exported same-shape real continuation bundle: {artifact}")
        return artifact
    finally:
        try:
            if env is not None:
                env.close()
        finally:
            simulation_app.close()


if __name__ == "__main__":
    main()
