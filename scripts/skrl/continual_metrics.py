"""Continual-learning metrics of finished sessions, computed from their logs (Isaac Sim is not needed).

Forward transfer (Wołczyk et al., 2021, "Continual World", arXiv:2105.10919, Eq. 2), from the success rate
of the training episodes logged in each stage's TensorBoard file:

    FT_i = (AUC_i - AUC_i^b) / (1 - AUC_i^b)

``AUC_i`` is the mean success rate over the training of task i (the area under the curve divided by the
training length), and ``AUC_i^b`` is the same for the baseline session (``scratch``: a new policy per task).
FT_i > 0: the method learns task i faster or better than a new policy; FT_i <= 1, it can be negative.
The success rate of a logged point is ``success / (crashed + lost + time_out)`` of the episodes that ended
in its write interval. An episode that is both crashed and lost counts twice in the denominator, so the
rate is a lower bound. The training episodes use the stochastic policy (with action noise).

``ft_after_first_episode`` leaves out the points that end within the first episode length of a stage:
at the start of a stage, the environment gives each episode a random elapsed time
(``spread_episode_starts``), so those episodes are shorter than a full one and pass more easily.

Cost (Díaz-Rodríguez et al., 2018, "Don't forget, there is more than forgetting", arXiv:1810.13166):
parameters, memory and FLOPs per action of the policy after each stage (``crazysim.cost``), the model size
efficiency ``MS = min(1, mean_i Mem(theta_1) / Mem(theta_i))`` (1: no growth), the number of stored states
(FAME's buffer of earlier tasks' states) and the CPU latency of one action.

Run with the Isaac Lab Python environment (it has ``tensorboard`` and ``skrl``, needed to load the policies):

    python scripts/skrl/continual_metrics.py --sessions logs/skrl/crazyflie_tracking_waypoint/<session> ... \\
        --output docs/continual_metrics_waypoint.json

The ``scratch`` session among ``--sessions`` is the baseline, unless ``--baseline`` is given.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import types

import numpy as np
import torch
import yaml
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Load crazysim.methods and crazysim.cost without importing the crazysim package (it registers the Isaac Lab
# environments, which needs Isaac Sim). The policies are pickled modules of crazysim.methods.
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_package = types.ModuleType("crazysim")
_package.__path__ = [os.path.join(REPO, "source", "crazysim", "crazysim")]
sys.modules.setdefault("crazysim", _package)
# crazysim.methods.base imports SkrlVecEnvWrapper for type hints only; its module imports Isaac Sim
_skrl_wrapper = types.ModuleType("isaaclab_rl.skrl")
_skrl_wrapper.SkrlVecEnvWrapper = object
sys.modules.setdefault("isaaclab_rl.skrl", _skrl_wrapper)

from crazysim.cost import cpu_latency_us, policy_cost  # noqa: E402

TERMINATION = "Info / Episode_Termination/"


class _ParamsLoader(yaml.SafeLoader):
    """Safe loader that reads the Python tags of Isaac Lab's ``dump_yaml`` (such as tuples) as plain data."""


def _plain(loader: yaml.SafeLoader, suffix: str, node: yaml.Node):
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node, deep=True)
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node, deep=True)
    return loader.construct_scalar(node)


_ParamsLoader.add_multi_constructor("tag:yaml.org,2002:python/", _plain)


def load_session(session_dir: str) -> dict:
    with open(os.path.join(session_dir, "session.json")) as f:
        session = json.load(f)
    with open(os.path.join(session_dir, "params", "env.yaml")) as f:
        env = yaml.load(f, Loader=_ParamsLoader)
    with open(os.path.join(session_dir, "params", "agent.yaml")) as f:
        session["agent_cfg"] = yaml.load(f, Loader=_ParamsLoader)
    step_dt = env["sim"]["dt"] * env["decimation"]
    # episode length of the session's first task; a fallback when TensorBoard has no episode lengths
    session["max_episode_length"] = math.ceil(env["episode_length_s"] / step_dt)
    session["obs_dim"] = env["observation_space"]
    session["dir"] = session_dir
    return session


def scalars(stage_dir: str, tag: str) -> dict[int, float]:
    """``{step: value}`` of a tag over all event files of a stage (later files override earlier ones)."""
    values = {}
    for path in sorted(glob.glob(os.path.join(stage_dir, "events.out.tfevents.*"))):
        accumulator = EventAccumulator(path, size_guidance={"scalars": 0})
        accumulator.Reload()
        if tag in accumulator.Tags()["scalars"]:
            values.update({event.step: event.value for event in accumulator.Scalars(tag)})
    return values


def success_curve(stage_dir: str) -> tuple[np.ndarray, np.ndarray]:
    """Steps and success rate of the training episodes; points where no episode ended are left out."""
    counts = {key: scalars(stage_dir, TERMINATION + key) for key in ("success", "crashed", "lost", "time_out")}
    steps, rates = [], []
    for step in sorted(counts["success"]):
        ended = sum(counts[key].get(step, 0.0) for key in ("crashed", "lost", "time_out"))
        if ended > 0:
            steps.append(step)
            rates.append(counts["success"][step] / ended)
    return np.array(steps), np.array(rates)


def stage_dir(stage: dict) -> str:
    # the stage directory of the policy: a resumed session lists stages of the session it continued
    return os.path.dirname(stage["policy"])


def learning_areas(session: dict) -> dict[str, dict]:
    """Per task: AUC over the whole training and after the first episode length."""
    areas = {}
    for stage in session["stages"]:
        steps, rates = success_curve(stage_dir(stage))
        if len(steps) == 0:
            raise ValueError(f"No success counts in {stage_dir(stage)}")
        interval = int(np.min(np.diff(steps))) if len(steps) > 1 else int(steps[0])
        # the episode length depends on the task (two laps of its reference): the longest logged episode
        lengths = scalars(stage_dir(stage), "Episode / Total timesteps (max)")
        episode_length = max(lengths.values()) if lengths else session["max_episode_length"]
        # a point averages the steps (step - interval, step]; keep it if they all follow the first episode
        late = steps - interval >= episode_length
        areas[stage["task"]] = {
            "auc": float(rates.mean()),
            "auc_after_first_episode": float(rates[late].mean()),
            "points": int(len(steps)),
            "points_after_first_episode": int(late.sum()),
            "episode_length": int(episode_length),
        }
    return areas


def forward_transfer(auc: float, baseline_auc: float) -> float:
    return (auc - baseline_auc) / (1.0 - baseline_auc) if baseline_auc < 1.0 else float("nan")


def stage_costs(session: dict, policy_file: str, latency: bool) -> list[dict]:
    costs = []
    for stage in session["stages"]:
        policy = torch.load(os.path.join(stage_dir(stage), policy_file), map_location="cpu", weights_only=False)
        cost = policy_cost(policy, session["obs_dim"])
        if latency:
            cost["cpu_latency_us"] = cpu_latency_us(policy, session["obs_dim"])
        costs.append(cost)
    return costs


def model_size_efficiency(costs: list[dict]) -> float:
    return min(1.0, float(np.mean([costs[0]["bytes"] / cost["bytes"] for cost in costs])))


def stored_states(session: dict) -> list[int]:
    """States kept from earlier tasks after each stage (FAME's buffer; 0 for the other methods)."""
    if session["method"] != "fame":
        return [0] * len(session["stages"])
    cfg = session["agent_cfg"]["methods"]["fame"]
    recorded = [stage.get("stored_states") for stage in session["stages"]]
    # sessions before the entry was recorded: every stage adds size_fast2meta states to a FIFO of size_meta
    return [
        count if count is not None else min(cfg["size_fast2meta"] * (index + 1), cfg["size_meta"])
        for index, count in enumerate(recorded)
    ]


def markdown_table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--sessions", nargs="+", required=True, help="Session directories, one per method.")
    parser.add_argument("--baseline", default=None, help="Baseline session (default: the scratch session).")
    parser.add_argument("--no_latency", action="store_true", help="Skip the CPU latency measurement.")
    parser.add_argument("--output", default=None, help="JSON file for all results.")
    args = parser.parse_args()

    sessions = [load_session(path) for path in args.sessions]
    if args.baseline:
        baseline = load_session(args.baseline)
    else:
        scratch = [session for session in sessions if session["method"] == "scratch"]
        if len(scratch) != 1:
            raise ValueError("Give --baseline, or exactly one scratch session in --sessions.")
        baseline = scratch[0]
    baseline_areas = learning_areas(baseline)

    results = {"baseline": baseline["dir"], "methods": {}}
    for session in sessions:
        tasks = [stage["task"] for stage in session["stages"]]
        areas = learning_areas(session)
        transfer = {}
        for task in tasks:
            if task not in baseline_areas:
                raise ValueError(f"The baseline session has no stage on {task}")
            transfer[task] = {
                "ft": forward_transfer(areas[task]["auc"], baseline_areas[task]["auc"]),
                "ft_after_first_episode": forward_transfer(
                    areas[task]["auc_after_first_episode"], baseline_areas[task]["auc_after_first_episode"]
                ),
            }
        costs = {"policy": stage_costs(session, "policy.pt", not args.no_latency)}
        if session["method"] == "fame":
            costs["meta_policy"] = stage_costs(session, "meta_policy.pt", not args.no_latency)
        storage = stored_states(session)
        results["methods"][session["method"]] = {
            "session": session["dir"],
            "tasks": tasks,
            "success_auc": areas,
            "forward_transfer": transfer,
            "mean_ft": float(np.mean([transfer[task]["ft"] for task in tasks])),
            "mean_ft_after_first_episode": float(np.mean([transfer[task]["ft_after_first_episode"] for task in tasks])),
            "cost": {
                name: {"stages": stage_list, "model_size_efficiency": model_size_efficiency(stage_list)}
                for name, stage_list in costs.items()
            },
            "stored_states": storage,
            "stored_bytes": [count * session["obs_dim"] * 4 for count in storage],  # float32 buffer
        }

    # tables
    tasks = results["methods"][sessions[0]["method"]]["tasks"]
    for key, title in (("ft", "Forward transfer (success rate, all points)"),
                       ("ft_after_first_episode", "Forward transfer (success rate, after the first episode length)")):
        rows = []
        for method, result in results["methods"].items():
            values = [result["forward_transfer"].get(task, {}).get(key, float("nan")) for task in tasks]
            mean = result["mean_ft" if key == "ft" else "mean_ft_after_first_episode"]
            rows.append([method] + [f"{value:.2f}" for value in values] + [f"{mean:.2f}"])
        print(f"\n{title}\n" + markdown_table(["method"] + tasks + ["mean"], rows))
    rows = []
    for method, result in results["methods"].items():
        rows.append([method] + [f"{result['success_auc'][task]['auc']:.3f}" for task in tasks])
    print("\nSuccess-rate AUC during training\n" + markdown_table(["method"] + tasks, rows))
    rows = []
    for method, result in results["methods"].items():
        for name, cost in result["cost"].items():
            last = cost["stages"][-1]
            label = method if name == "policy" else f"{method} (meta)"
            rows.append([
                label,
                " / ".join(str(stage["parameters"]) for stage in cost["stages"]),
                " / ".join(str(stage.get("trainable_parameters", "-")) for stage in cost["stages"]),
                f"{last['bytes'] / 1024:.0f}",
                f"{cost['model_size_efficiency']:.2f}",
                f"{last['flops_per_action'] / 1000:.0f}",
                f"{last['cpu_latency_us']:.0f}" if "cpu_latency_us" in last else "-",
                f"{result['stored_states'][-1]} ({result['stored_bytes'][-1] / 2**20:.0f} MiB)",
            ])
    print("\nCost\n" + markdown_table(
        ["method", "parameters after each stage", "trainable parameters in each stage", "final memory [KiB]", "MS", "final kFLOPs per action",
         "final CPU latency [µs]", "stored states after the last stage"], rows))

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n[INFO] Results written to {args.output}")


if __name__ == "__main__":
    main()
