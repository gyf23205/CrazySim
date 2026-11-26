"""Analysis figures: methods compared on each new task, and the inside of the KAE-MoE policy (Isaac Sim is not needed).

Inputs: the JSON of ``scripts/skrl/continual_metrics.py`` (sessions, transfer matrices, training logs) and the output
directory of ``scripts/skrl/analysis_rollouts.py`` (``rollouts.json``, ``traces.npz``).

Figures (PNG and PDF in ``--output_dir``):

- ``start_vs_final``: position error on each new task with the policy a method starts the stage with (jumpstart)
  and with the policy it ends the stage with.
- ``time_to_success``: training steps until the success rate of the training episodes first reaches a threshold.
- ``flight_paths``: flight path of each method's stage policy on its own task, against the reference.
- ``trainable_parameters``: parameters trained in each stage.
- ``kae_gate_ablation``: KAE-MoE with the learned gate, the MLP pathway only and the KAE pathway only.
- ``kae_gate_on_path``: KAE-MoE flight paths colored by the gate value.
- ``kae_router``: share of the KAE pathway each earlier task's KAE receives.
- ``kae_spectrum``: eigenvalues of each KAE's Koopman matrix, sized by how much the policy uses each mode.
- ``kae_fidelity``: actions of each KAE against the actions of the policy it was fitted to.

Example:
    python scripts/skrl/plot_analysis.py --metrics docs/figures/waypoint_5tasks_1/metrics.json \
        --rollouts docs/figures/waypoint_5tasks_1/analysis --output_dir docs/figures/waypoint_5tasks_1/analysis
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plot_continual_metrics as pcm  # noqa: E402  (also sets the matplotlib style)
from continual_metrics import success_curve, stage_dir  # noqa: E402
from plot_continual_metrics import (  # noqa: E402
    AXIS, BLUES, GRID, INK, INK_SECONDARY, METHODS, MUTED, SURFACE, color, ink_on, label, marker, save,
)

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

CRASH_LIMIT = 0.5
"""Crash rate from which a result is marked as crashed (as in the transfer matrices)."""
SUCCESS_THRESHOLDS = (0.5, 0.9)
# action dimensions: categorical slots 1-3 of the validated palette (blue, orange, aqua)
ACTION_COLORS = ("#2a78d6", "#eb6834", "#1baf7a")
ACTION_NAMES = ("x", "y", "z")
# gate ablation: the learned gate in the KAE-MoE color, the two fixed gates in the reference gray, told apart by hatch
ABLATION = (("learned", "learned gate", METHODS["kae_moe"][1], ""),
            ("mlp_only", "g = 0 (MLP pathway only)", "#c3c2b7", ""),
            ("kae_only", "g = 1 (KAE pathway only)", "#898781", "////"))


def short(task: str) -> str:
    return task.replace("vertical_circle", "v_circle")


def load_sessions(metrics: dict) -> dict[str, dict]:
    sessions = {}
    for method, result in metrics["methods"].items():
        with open(os.path.join(result["session"], "session.json")) as f:
            sessions[method] = json.load(f)
    return sessions


def reference_of(traces: dict, key: str, laps: int = 2) -> np.ndarray:
    """First lap of the reference of a trace (later laps overlap it and fill the gaps of a dashed line).

    The first sample is left out: it is read before the reset updates the reference, so it is the last reference
    point of the previous run. ``laps``: laps per episode (``trajectory_laps`` of the environment).
    """
    reference = traces[key][1:]
    return reference[: len(reference) // laps + 2]


def plane_axes(reference: np.ndarray) -> tuple[int, int]:
    """The two coordinates with the largest range of the reference (the plane of the trajectory)."""
    spans = reference.max(axis=0) - reference.min(axis=0)
    return tuple(sorted(np.argsort(spans)[-2:]))


def style_path_axes(ax, axes: tuple[int, int]):
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel(f"{'xyz'[axes[0]]} [m]")
    ax.set_ylabel(f"{'xyz'[axes[1]]} [m]")
    ax.grid(True)


# -- comparison of the methods ----------------------------------------------------------------------------------------
def plot_start_vs_final(metrics: dict, rollouts: dict, tasks: list[str], output_dir: str):
    """Dumbbell per method and task: error of the starting policy (hollow) and of the final policy (filled)."""
    matrices = pcm.transfer_matrices(metrics, tasks)
    methods = [m for m in METHODS if m in rollouts["jumpstart"]]
    new_tasks = tasks[1:]  # stage 0 starts every method from the same random policy
    fig, axes = plt.subplots(1, len(new_tasks), figsize=(3.3 * len(new_tasks), 0.45 * len(methods) + 1.6),
                             sharey=True)
    y = np.arange(len(methods))[::-1]
    for ax, task in zip(axes, new_tasks):
        column = tasks.index(task)
        for row, method in zip(y, methods):
            if task not in rollouts["jumpstart"][method]:
                continue  # not evaluated (yet)
            start = rollouts["jumpstart"][method][task]
            start_error, start_crash = start["position_error"] * 100.0, start["crash_rate"]
            final_error = matrices[method]["error"][column, column]
            final_crash = matrices[method]["crash"][column, column]
            ax.plot([start_error, final_error], [row, row], color=AXIS, linewidth=2, zorder=1)
            ax.scatter(start_error, row, s=46, facecolor=SURFACE, edgecolor=color(method), linewidth=1.6,
                       marker=marker(method), zorder=2)
            ax.scatter(final_error, row, s=52, facecolor=color(method), edgecolor=SURFACE, linewidth=0.8,
                       marker=marker(method), zorder=3)
            # crash label beside its marker, on the side away from the connecting line
            right = start_error >= final_error
            for error, crash, outside in ((start_error, start_crash, right), (final_error, final_crash, not right)):
                if crash >= CRASH_LIMIT:
                    ax.annotate(f"{crash * 100:.0f}%\ncrash", (error, row), xytext=(7 if outside else -7, 0),
                                textcoords="offset points", ha="left" if outside else "right", va="center",
                                fontsize=6.5, color=INK_SECONDARY, linespacing=0.9)
        ax.set_title(task, loc="left")
        ax.set_xlim(0, ax.get_xlim()[1] + 9)
        ax.set_xlabel("position error [cm]")
        ax.grid(axis="x")
    axes[0].set_yticks(y, [label(m) for m in methods])
    axes[0].set_ylim(-0.7, len(methods) - 0.3)
    handles = [Line2D([], [], linestyle="none", marker="o", markersize=7, markerfacecolor=SURFACE,
                      markeredgecolor=INK_SECONDARY, label="start of the stage (before training)"),
               Line2D([], [], linestyle="none", marker="o", markersize=7, markerfacecolor=INK_SECONDARY,
                      markeredgecolor=SURFACE, label="end of the stage (selected policy)")]
    fig.legend(handles=handles, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("Error on each new task at the start and at the end of its stage (lower is better)",
                 x=0.01, y=1.07, ha="left", fontsize=11)
    fig.text(0.01, -0.04, f"Start: the policy each method begins the stage with, evaluated without training "
             f"({rollouts['num_envs']} episodes from random points of the reference). End: the transfer matrix "
             f"diagonal. Error of a crashed episode covers the steps before the crash; labels mark crash rates "
             f">= {CRASH_LIMIT * 100:.0f}%.", fontsize=7.5, color=INK_SECONDARY)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, output_dir, "start_vs_final")


def first_success_step(directory: str, threshold: float, skip_steps: int) -> float:
    """First training step at which the success rate reaches ``threshold`` (NaN if never)."""
    steps, rates = success_curve(directory)
    keep = steps > skip_steps
    hits = steps[keep][rates[keep] >= threshold]
    return float(hits[0]) if len(hits) else float("nan")


def plot_time_to_success(metrics: dict, sessions: dict, tasks: list[str], output_dir: str):
    methods = [m for m in METHODS if m in metrics["methods"]]
    fig, axes = plt.subplots(1, len(SUCCESS_THRESHOLDS),
                             figsize=(0.95 * len(tasks) * len(SUCCESS_THRESHOLDS) + 2.6, 0.42 * len(methods) + 1.6))
    timesteps = None
    for ax, threshold in zip(np.atleast_1d(axes), SUCCESS_THRESHOLDS):
        values = np.full((len(methods), len(tasks)), np.nan)
        for r, method in enumerate(methods):
            stages = {stage["task"]: stage for stage in sessions[method]["stages"]}
            for c, task in enumerate(tasks):
                # points within the first episode length are left out: episodes start at random points of the
                # reference, so the first episodes to end are short ones (as for the forward transfer)
                skip = metrics["methods"][method]["success_auc"][task]["episode_length"]
                values[r, c] = first_success_step(stage_dir(stages[task]), threshold, skip)
                timesteps = timesteps or sessions[method].get("timesteps")
        vmax = 48000.0
        ax.imshow(np.where(np.isnan(values), np.nan, values), cmap=BLUES.reversed(), vmin=0, vmax=vmax,
                  aspect="auto")
        for r in range(len(methods)):
            for c in range(len(tasks)):
                value = values[r, c]
                if np.isnan(value):
                    ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor=SURFACE, edgecolor=GRID,
                                               hatch="///", linewidth=0))
                    ax.text(c, r, "never", ha="center", va="center", fontsize=7.5, color=INK_SECONDARY)
                else:
                    fill = BLUES.reversed()(value / vmax)
                    ax.text(c, r, f"{value / 1000:.1f}K", ha="center", va="center", fontsize=8, color=ink_on(fill))
        ax.set_xticks(range(len(tasks)), [short(t) for t in tasks], rotation=30, ha="right")
        ax.set_yticks(range(len(methods)), [label(m) for m in methods])
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(f"success rate >= {threshold * 100:.0f}%", loc="left")
    fig.suptitle("Training steps until the success rate first reaches a threshold (of 48K; darker is faster)",
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    save(fig, output_dir, "time_to_success")


def plot_flight_paths(rollouts: dict, traces: dict, tasks: list[str], output_dir: str):
    methods = [m for m in METHODS if m in rollouts["traces"]]
    fig, axes = plt.subplots(len(methods), len(tasks), figsize=(2.5 * len(tasks), 2.35 * len(methods)))
    for c, task in enumerate(tasks):
        reference = reference_of(traces, f"{methods[0]}/{task}/reference")
        plane = plane_axes(reference)
        for r, method in enumerate(methods):
            ax = axes[r, c]
            path = traces[f"{method}/{task}/position"]
            ref = reference_of(traces, f"{method}/{task}/reference")
            ax.plot(ref[:, plane[0]], ref[:, plane[1]], color=MUTED, linewidth=1.0, linestyle=(0, (3, 2)))
            ax.plot(path[:, plane[0]], path[:, plane[1]], color=color(method), linewidth=1.3)
            ax.scatter(*path[0, list(plane)], s=14, color=INK_SECONDARY, zorder=3)
            info = rollouts["traces"][method][task]
            if info["crashed_env0"]:
                ax.scatter(*path[-1, list(plane)], s=40, marker="x", color=INK, linewidth=1.5, zorder=4)
            ax.text(0.02, 0.98, f"{info['position_error'] * 100:.1f} cm"
                    + (f", {info['crash_rate'] * 100:.0f}% crash" if info["crash_rate"] > 0 else ""),
                    transform=ax.transAxes, va="top", fontsize=7, color=INK_SECONDARY)
            style_path_axes(ax, plane)
            ax.tick_params(labelsize=6.5)
            ax.xaxis.label.set_size(7)
            ax.yaxis.label.set_size(7)
            if c > 0:
                ax.set_ylabel("")
            if r < len(methods) - 1:
                ax.set_xlabel("")
            if r == 0:
                ax.set_title(task, loc="left")
            if c == 0:
                ax.annotate(label(method), (-0.42, 0.5), xycoords="axes fraction", rotation=90, ha="center",
                            va="center", fontsize=9.5)
    handles = [Line2D([], [], color=MUTED, linestyle=(0, (3, 2)), label="reference"),
               Line2D([], [], color=INK_SECONDARY, linestyle="none", marker="o", markersize=4, label="start"),
               Line2D([], [], color=INK, linestyle="none", marker="x", markersize=6, label="crash")]
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("Flight path of each stage's policy on its own task (one episode from the first reference point)",
                 x=0.01, y=1.012, ha="left", fontsize=11)
    fig.text(0.01, -0.005, f"Text: mean position error and crash rate over {rollouts['num_envs']} episodes of the same "
             "run (spawn positions differ by up to 0.1 m).", fontsize=7.5, color=INK_SECONDARY)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    save(fig, output_dir, "flight_paths")


def plot_trainable_parameters(metrics: dict, tasks: list[str], output_dir: str):
    series = {}
    for method, result in metrics["methods"].items():
        series[method] = [stage["trainable_parameters"] for stage in result["cost"]["policy"]["stages"]]
    # FAME's meta learner (trained by distillation, not RL) is the same size as the MLP: a note, not a line
    meta = [result["cost"]["meta_policy"]["stages"] for result in metrics["methods"].values()
            if "meta_policy" in result["cost"]]
    methods = [m for m in METHODS if m in series]
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    x = np.arange(len(tasks))
    # identical series are drawn once, labelled with every method they belong to
    groups: dict[tuple, list[str]] = {}
    for method in methods:
        groups.setdefault(tuple(series[method]), []).append(method)
    for values, members in groups.items():
        line_color = color(members[0]) if len(members) == 1 else MUTED
        name = ", ".join(label(m) for m in members)
        ax.plot(x, values, color=line_color, marker=marker(members[0]) if len(members) == 1 else "o", markersize=6,
                markeredgecolor=SURFACE, markeredgewidth=0.8, label=name)
        ax.annotate(f"{values[-1] / 1000:.1f}K", (x[-1], values[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=8, color=INK_SECONDARY)
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_formatter(pcm.THOUSANDS)
    ax.set_xticks(x, [f"{i}: {short(t)}" for i, t in enumerate(tasks)])
    ax.set_xlabel("stage")
    ax.set_ylabel("trainable parameters")
    ax.grid(axis="y")
    ax.legend(loc="upper left", bbox_to_anchor=(1.07, 1.0))
    if meta:
        sizes = sorted({stage["trainable_parameters"] for stage in meta[0]})
        ax.text(1.09, 0.45, f"fame (meta), trained by distillation:\n{', '.join(f'{v:,}' for v in sizes)} "
                "in every stage", transform=ax.transAxes, fontsize=8, color=INK_SECONDARY, va="top")
    ax.set_title("Parameters trained in each stage (policy network; frozen modules of earlier stages excluded)",
                 loc="left")
    fig.tight_layout()
    save(fig, output_dir, "trainable_parameters")


# -- KAE-MoE ------------------------------------------------------------------------------------------------------------
def plot_gate_ablation(rollouts: dict, tasks: list[str], output_dir: str):
    ablation = rollouts["ablation"]
    moe_tasks = [t for t in tasks if t in ablation]
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.4))
    width = 0.26
    x = np.arange(len(moe_tasks))
    for ax, (key, ylabel, scale) in zip(axes, (("position_error", "position error [cm]", 100.0),
                                                ("crash_rate", "crash rate [%]", 100.0))):
        for k, (name, text, fill, hatch) in enumerate(ABLATION):
            values = [ablation[t][name][key] * scale for t in moe_tasks]
            bars = ax.bar(x + (k - 1) * width, values, width - 0.03, color=fill, hatch=hatch, edgecolor=SURFACE,
                          linewidth=0, label=text)
            for bar, value in zip(bars, values):
                ax.annotate(f"{value:.0f}" if value >= 10 else f"{value:.1f}",
                            (bar.get_x() + bar.get_width() / 2, value), xytext=(0, 2), textcoords="offset points",
                            ha="center", fontsize=7, color=INK_SECONDARY)
        ax.set_xticks(x, moe_tasks)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y")
        ax.set_ylim(bottom=0, top=105 if key == "crash_rate" else None)
    axes[0].set_title("Position error (lower is better)", loc="left")
    axes[1].set_title("Crash rate (lower is better)", loc="left")
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("KAE-MoE with the gate fixed: how much each pathway contributes (each stage's policy on its task)",
                 x=0.01, y=1.06, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save(fig, output_dir, "kae_gate_ablation")


def plot_gate_on_path(rollouts: dict, traces: dict, tasks: list[str], output_dir: str):
    moe_tasks = [t for t in tasks[1:] if f"kae_moe/{t}/gate" in traces]
    fig, axes = plt.subplots(1, len(moe_tasks), figsize=(3.4 * len(moe_tasks) + 0.8, 3.6), layout="constrained")
    norm = plt.Normalize(0.0, 1.0)
    for ax, task in zip(axes, moe_tasks):
        path = traces[f"kae_moe/{task}/position"]
        reference = reference_of(traces, f"kae_moe/{task}/reference")
        gate = traces[f"kae_moe/{task}/gate"]
        plane = plane_axes(reference)
        ax.plot(reference[:, plane[0]], reference[:, plane[1]], color=MUTED, linewidth=1.0, linestyle=(0, (3, 2)))
        points = path[:, list(plane)]
        segments = np.stack([points[:-1], points[1:]], axis=1)
        lines = LineCollection(segments, cmap=BLUES, norm=norm, linewidth=2.2, capstyle="round")
        lines.set_array(gate[:-1])
        ax.add_collection(lines)
        ax.scatter(*points[0], s=14, color=INK_SECONDARY, zorder=3)
        style_path_axes(ax, plane)
        ax.autoscale_view()
        ax.set_title(f"stage {tasks.index(task)}: {task}\ngate mean {gate.mean():.2f}, range {gate.min():.2f}-"
                     f"{gate.max():.2f}", loc="left")
    colorbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=BLUES), ax=axes, fraction=0.025, pad=0.02)
    colorbar.set_label("gate g  (1: KAE pathway, 0: MLP pathway)")
    colorbar.outline.set_visible(False)
    fig.suptitle("Where along the path KAE-MoE relies on the KAE pathway (one episode, dashed: reference)",
                 x=0.01, ha="left", fontsize=11)
    save(fig, output_dir, "kae_gate_on_path")


def plot_router(rollouts: dict, traces: dict, tasks: list[str], output_dir: str):
    """Mean router share and mean |mode weight| per source KAE, for each stage that uses the KAEs."""
    moe_tasks = [t for t in tasks[1:] if f"kae_moe/{t}/mean_router" in traces]
    sources = tasks[: len(moe_tasks)]
    router = np.full((len(moe_tasks), len(sources)), np.nan)
    for r, task in enumerate(moe_tasks):
        values = traces[f"kae_moe/{task}/mean_router"]
        router[r, : len(values)] = values
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.4), gridspec_kw={"width_ratios": [1.0, 1.25]})
    ax = axes[0]
    ax.imshow(router, cmap=BLUES, vmin=0, vmax=1, aspect="auto")
    for r in range(len(moe_tasks)):
        for c in range(len(sources)):
            if np.isnan(router[r, c]):
                ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor=SURFACE, edgecolor=GRID, hatch="///",
                                           linewidth=0))
            else:
                ax.text(c, r, f"{router[r, c]:.2f}", ha="center", va="center", fontsize=8.5,
                        color=ink_on(BLUES(router[r, c])))
    ax.set_xticks(range(len(sources)), [short(t) for t in sources])
    ax.set_yticks(range(len(moe_tasks)), [f"{tasks.index(t)}: {short(t)}" for t in moe_tasks])
    ax.set_xlabel("KAE fitted on task")
    ax.set_ylabel("stage (task being learned)")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("Router share of each KAE (rows sum to 1)", loc="left")

    # router share over one episode: one line per source KAE, one panel would mix stages, so use the last stage
    ax = axes[1]
    task = moe_tasks[-1]
    trace = traces[f"kae_moe/{task}/router"]
    reference = traces[f"kae_moe/{task}/reference"]
    time = np.arange(len(trace)) / len(trace) * 100.0
    source_colors = [METHODS[m][1] for m in ("finetune", "finetune_reset", "residual", "componet")]
    for k, source in enumerate(sources):
        ax.plot(time, trace[:, k], color=source_colors[k], label=f"KAE of {short(source)}")
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 100)
    ax.set_xlabel("episode progress [%]")
    ax.set_ylabel("router share")
    ax.grid(axis="y")
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    ax.set_title(f"Router share during one episode of stage {tasks.index(task)} ({short(task)})", loc="left")
    fig.suptitle("Which earlier skills KAE-MoE reuses (means over all environments and steps of one episode each)",
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    save(fig, output_dir, "kae_router")


def plot_spectrum(rollouts: dict, traces: dict, tasks: list[str], output_dir: str):
    """Eigenvalues of each KAE's Koopman matrix; marker area: mean |mode weight| over the stages that use the KAE."""
    fidelity = rollouts["kae_fidelity"]
    moe_tasks = [t for t in tasks[1:] if f"kae_moe/{t}/mean_abs_weights" in traces]
    kaes = [t for t in tasks if t in fidelity]
    largest = max(traces[f"kae_moe/{t}/mean_abs_weights"].max() for t in moe_tasks)
    fig, axes = plt.subplots(1, len(kaes), figsize=(2.9 * len(kaes), 3.3))
    for k, (ax, source) in enumerate(zip(axes, kaes)):
        eig = np.array(fidelity[source]["eigvals_p_real"]) + 1j * np.array(fidelity[source]["eigvals_p_imag"])
        modes = len(eig)
        # mode weights of this KAE in every later stage: entries k*modes:(k+1)*modes of the weight vector
        usage = [traces[f"kae_moe/{t}/mean_abs_weights"][k * modes:(k + 1) * modes] for t in moe_tasks
                 if len(traces[f"kae_moe/{t}/mean_abs_weights"]) >= (k + 1) * modes]
        theta = np.linspace(0, 2 * np.pi, 200)
        ax.plot(np.cos(theta), np.sin(theta), color=AXIS, linewidth=1.0)
        ax.axhline(0, color=GRID, linewidth=0.8, zorder=0)
        ax.axvline(0, color=GRID, linewidth=0.8, zorder=0)
        if usage:
            weight = np.mean(usage, axis=0)
            sizes = 10 + 500 * weight / largest
            ax.scatter(eig.real, eig.imag, s=sizes, color=METHODS["kae_moe"][1], alpha=0.7, edgecolor=SURFACE,
                       linewidth=0.8, zorder=3)
            first, last = tasks.index(source) + 1, len(tasks) - 1
            note = f"used by stage{'s' if last > first else ''} {first}" + (f"-{last}" if last > first else "")
        else:
            ax.scatter(eig.real, eig.imag, s=20, facecolor=SURFACE, edgecolor=INK_SECONDARY, linewidth=1.0, zorder=3)
            note = "not used (last task)"
        radius = np.abs(eig).max()
        ax.set_aspect("equal")
        limit = max(1.1, radius * 1.1)
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.set_title(f"KAE of {source}\n{note}, ρ = {radius:.2f}", loc="left")
        ax.set_xlabel("Re(λ)")
        if k == 0:
            ax.set_ylabel("Im(λ)")
    fig.suptitle("Koopman eigenvalues λ of each KAE (circle: |λ| = 1; ρ: spectral radius, the largest |λ|; marker area: "
                 "mean |weight| the later policies give the mode, one scale for all panels)", x=0.01, ha="left",
                 fontsize=10.5)
    fig.tight_layout()
    save(fig, output_dir, "kae_spectrum")


def plot_fidelity(rollouts: dict, traces: dict, tasks: list[str], output_dir: str):
    fidelity = rollouts["kae_fidelity"]
    kaes = [t for t in tasks if t in fidelity]
    fig, axes = plt.subplots(1, len(kaes), figsize=(2.9 * len(kaes), 3.3))
    for ax, source in zip(axes, kaes):
        target = traces[f"kae_fidelity/{source}/policy"]
        prediction = traces[f"kae_fidelity/{source}/kae"]
        low = min(target.min(), prediction.min())
        high = max(target.max(), prediction.max())
        ax.plot([low, high], [low, high], color=AXIS, linewidth=1.0, zorder=0)
        for d in range(target.shape[1]):
            r2 = fidelity[source]["r2_per_action"][d]
            ax.scatter(target[:, d], prediction[:, d], s=3, color=ACTION_COLORS[d], alpha=0.35, linewidth=0,
                       label=f"{ACTION_NAMES[d]}: R² {r2:.3f}")
        ax.set_aspect("equal")
        ax.set_title(f"KAE of {source}", loc="left")
        ax.set_xlabel("policy action")
        ax.legend(loc="upper left", markerscale=4, handletextpad=0.2, fontsize=7.5)
    axes[0].set_ylabel("KAE action")
    fig.suptitle("Does each KAE reproduce the policy it was fitted to? (states visited by that policy on its task)",
                 x=0.01, ha="left", fontsize=11)
    fig.text(0.01, -0.02, "R² = 1 - MSE / variance of the policy action, per action dimension (waypoint offset x, y, "
             "z); 1 is a perfect copy.", fontsize=7.5, color=INK_SECONDARY)
    fig.tight_layout()
    save(fig, output_dir, "kae_fidelity")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--metrics", required=True, help="JSON written by scripts/skrl/continual_metrics.py.")
    parser.add_argument("--rollouts", required=True, help="Output directory of scripts/skrl/analysis_rollouts.py.")
    parser.add_argument("--output_dir", required=True, help="Directory for the figures.")
    args = parser.parse_args()
    with open(args.metrics) as f:
        metrics = json.load(f)
    with open(os.path.join(args.rollouts, "rollouts.json")) as f:
        rollouts = json.load(f)
    traces_path = os.path.join(args.rollouts, "traces.npz")
    traces = dict(np.load(traces_path)) if os.path.isfile(traces_path) else {}
    os.makedirs(args.output_dir, exist_ok=True)
    tasks = rollouts["tasks"]
    sessions = load_sessions(metrics)

    plot_time_to_success(metrics, sessions, tasks, args.output_dir)
    plot_trainable_parameters(metrics, tasks, args.output_dir)
    if "jumpstart" in rollouts:
        plot_start_vs_final(metrics, rollouts, tasks, args.output_dir)
    if traces:
        plot_flight_paths(rollouts, traces, tasks, args.output_dir)
        plot_gate_on_path(rollouts, traces, tasks, args.output_dir)
        plot_router(rollouts, traces, tasks, args.output_dir)
        plot_spectrum(rollouts, traces, tasks, args.output_dir)
        plot_fidelity(rollouts, traces, tasks, args.output_dir)
    if "ablation" in rollouts:
        plot_gate_ablation(rollouts, tasks, args.output_dir)


if __name__ == "__main__":
    main()
