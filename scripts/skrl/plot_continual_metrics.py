"""Figures that compare the transfer methods of a continual-learning experiment (Isaac Sim is not needed).

Input: the JSON written by ``scripts/skrl/continual_metrics.py`` (it lists the sessions). Also read: each session's
``session.json`` (transfer matrix), TensorBoard files (training curves) and, for FAME, ``meta_policy_eval.json``
(the meta learner's transfer matrix, written by ``scripts/skrl/evaluate.py``; skipped if missing).

Figures (PNG and PDF in ``--output_dir``):

- ``summary``: one panel per metric, methods in the same rows: mean error and crash rate of the final policy over all
  tasks, forgetting, mean forward transfer, parameters of the final policy.
- ``forward_transfer``: forward transfer per method and task (diverging: blue above ``scratch``, red below).
- ``transfer_matrices``: error of the policy after each stage on each task, one panel per method; hatched cells
  crashed in at least half of the episodes.
- ``learning_curves``: training success rate and position error over the training of each task.
- ``reward_curves``: mean episode return and mean reward per step over the training of each task.
- ``cost``: parameters, FLOPs and CPU latency per action of the policy after each stage.

Forgetting of task i is ``error_i(after the last task) - error_i(after task i)`` [cm], averaged over all tasks but
the last (Wołczyk et al., 2021, Eq. 3, with error instead of success rate, so positive means forgetting). The error
of a crashed episode covers only the steps before the crash.

Example (Isaac Lab Python environment):

    python scripts/skrl/continual_metrics.py --sessions <session> ... --output metrics.json
    python scripts/skrl/plot_continual_metrics.py --metrics metrics.json --output_dir docs/figures/waypoint_5tasks
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm, to_rgb  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from continual_metrics import scalars, stage_dir, success_curve  # noqa: E402

# -- style ----------------------------------------------------------------------------------------------------------
# Categorical slots in fixed order (validated: adjacent CVD ΔE >= 9.1, normal-vision ΔE >= 19.6 on #fcfcfb).
# scratch is the reference, so it takes the muted gray instead of a slot. Every method also has its own marker,
# because line charts put every pair of series next to each other.
METHODS = {
    # name: (label, color, marker)
    "scratch": ("scratch", "#898781", "o"),
    "finetune": ("finetune", "#2a78d6", "s"),
    "finetune_reset": ("finetune_reset", "#eb6834", "D"),
    "residual": ("residual", "#1baf7a", "^"),
    "componet": ("componet", "#eda100", "v"),
    "kae_moe": ("kae_moe", "#e87ba4", "P"),
    "fame": ("fame (fast)", "#008300", "X"),
    "fame_meta": ("fame (meta)", "#6250d6", "*"),
}
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
HATCH_INK = (11 / 255, 11 / 255, 11 / 255, 0.45)
# sequential: one hue (blue), light to dark
BLUES = LinearSegmentedColormap.from_list("blues", [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
    "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"])
# diverging: red (below the reference) - neutral gray - blue (above)
DIVERGING = LinearSegmentedColormap.from_list("red_gray_blue", ["#e34948", "#f0efec", "#2a78d6"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif", "font.size": 9,
    "text.color": INK, "axes.labelcolor": INK_SECONDARY, "axes.titlecolor": INK, "axes.titlesize": 10,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_SECONDARY, "ytick.labelcolor": INK_SECONDARY,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": False, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
    "lines.linewidth": 1.5, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
    "legend.frameon": False, "legend.fontsize": 8.5,
})


def label(method: str) -> str:
    return METHODS[method][0]


def color(method: str) -> str:
    return METHODS[method][1]


def marker(method: str) -> str:
    return METHODS[method][2]


def ink_on(fill) -> str:
    """Primary ink or white, whichever reads on ``fill``."""
    r, g, b = to_rgb(fill)
    return INK if 0.2126 * r + 0.7152 * g + 0.0722 * b > 0.5 else "#ffffff"


def signed(value: float, digits: int = 2) -> str:
    """``+0.47`` / ``-1.23``; values that round to zero print as ``0.00`` (no ``-0.00``)."""
    if round(value, digits) == 0:
        return f"{0.0:.{digits}f}"
    return f"{value:+.{digits}f}"


THOUSANDS = matplotlib.ticker.FuncFormatter(lambda value, _: f"{value / 1000:.0f}K" if value >= 1000 else f"{value:.0f}")


def save(fig, output_dir: str, name: str):
    for extension in ("png", "pdf"):
        fig.savefig(os.path.join(output_dir, f"{name}.{extension}"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Wrote {os.path.join(output_dir, name)}.png/.pdf")


# -- data -----------------------------------------------------------------------------------------------------------
def transfer_matrices(metrics: dict, tasks: list[str]) -> dict[str, dict[str, np.ndarray]]:
    """Per method: ``error`` [cm] and ``crash`` [0, 1], rows: after stage i, columns: evaluated task j."""
    matrices = {}
    for method, result in metrics["methods"].items():
        with open(os.path.join(result["session"], "session.json")) as f:
            session = json.load(f)
        rows = [session["transfer"][stage["stage"]] for stage in session["stages"]]
        matrices[method] = matrix_from_rows(rows, tasks)
        meta_eval = os.path.join(result["session"], "meta_policy_eval.json")
        if method == "fame" and os.path.isfile(meta_eval):
            with open(meta_eval) as f:
                evaluation = json.load(f)
            # keys: "<stage>/meta_policy.pt"; sort by the stage index
            rows = [evaluation[key] for key in sorted(evaluation, key=lambda key: int(key.split("_")[0]))]
            matrices["fame_meta"] = matrix_from_rows(rows, tasks)
    return matrices


def matrix_from_rows(rows: list[dict], tasks: list[str]) -> dict[str, np.ndarray]:
    error = np.array([[row[task]["position_error"] * 100.0 for task in tasks] for row in rows])
    crash = np.array([[row[task]["crash_rate"] for task in tasks] for row in rows])
    return {"error": error, "crash": crash}


def summary_values(metrics: dict, matrices: dict) -> dict[str, dict[str, float]]:
    values = {}
    for method, matrix in matrices.items():
        error, crash = matrix["error"], matrix["crash"]
        n = error.shape[0]
        result = metrics["methods"].get("fame" if method == "fame_meta" else method)
        cost = result["cost"]["meta_policy" if method == "fame_meta" else "policy"]["stages"][-1]
        values[method] = {
            "final_error": float(error[-1].mean()),
            "final_crash": float(crash[-1].mean() * 100.0),
            "forgetting": float(np.mean([error[-1, i] - error[i, i] for i in range(n - 1)])),
            # the meta learner is distilled, not trained on the tasks: it has no learning curve
            "forward_transfer": float("nan") if method == "fame_meta" else result["mean_ft"],
            "parameters": float(cost["parameters"]),
        }
    return values


# -- figures --------------------------------------------------------------------------------------------------------
def plot_summary(values: dict, output_dir: str):
    methods = list(values)
    panels = [
        ("final_error", "Final policy: mean error [cm]", "{:.1f}", False),
        ("final_crash", "Final policy: crash rate [%]", "{:.0f}", False),
        ("forgetting", "Forgetting [cm of error]", "{:+.1f}", False),
        ("forward_transfer", "Forward transfer (mean)", "signed", False),
        ("parameters", "Parameters of the final policy", "{:,.0f}", False),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(15, 0.42 * len(methods) + 1.2), sharey=True)
    y = np.arange(len(methods))
    for ax, (key, title, fmt, log) in zip(axes, panels):
        data = np.array([values[m][key] for m in methods])
        ax.barh(y, np.nan_to_num(data), height=0.6, color=[color(m) for m in methods], zorder=2)
        ax.axvline(0, color=AXIS, linewidth=0.8, zorder=3)
        ax.set_title(title, loc="left")
        ax.grid(axis="x", zorder=0)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        if log:
            ax.set_xscale("log")
            ax.set_xlim(left=1e3)
        finite = data[np.isfinite(data)]
        span = max(np.abs(finite).max(), 1e-9)
        if not log:
            low = min(0.0, finite.min())
            ax.set_xlim(low - 0.08 * span if low < 0 else 0, max(0.0, finite.max()) + 0.3 * span)
        if key == "parameters":
            ax.xaxis.set_major_formatter(THOUSANDS)
        for yi, method, value in zip(y, methods, data):
            if key == "forward_transfer" and method == "scratch":
                ax.text(0, yi, "  baseline", va="center", ha="left", color=MUTED, fontsize=8)
                continue
            if not np.isfinite(value):
                ax.text(0, yi, "  n/a", va="center", ha="left", color=MUTED, fontsize=8)
                continue
            offset = value * 1.15 if log else value + (0.03 * span if value >= 0 else -0.03 * span)
            ha = "left" if value >= 0 or log else "right"
            text = signed(value) if fmt == "signed" else fmt.format(value)
            ax.text(offset, yi, text, va="center", ha=ha, color=INK_SECONDARY, fontsize=8)
    axes[0].set_yticks(y, [label(m) for m in methods])
    axes[0].invert_yaxis()
    fig.suptitle("Continual learning: comparison of the methods", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    save(fig, output_dir, "summary")


def plot_forward_transfer(metrics: dict, tasks: list[str], output_dir: str):
    # scratch is the baseline: its forward transfer is 0 by definition
    methods = [m for m in METHODS if m in metrics["methods"] and m != "scratch"]
    data = np.array([[metrics["methods"][m]["forward_transfer"][t]["ft"] for t in tasks] for m in methods])
    means = np.array([metrics["methods"][m]["mean_ft"] for m in methods])
    table = np.column_stack([data, means])
    limit = np.nanmax(np.abs(table))
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)  # equal arms: same color distance per unit
    fig, ax = plt.subplots(figsize=(7.5, 0.45 * len(methods) + 1.4))
    image = ax.imshow(table, cmap=DIVERGING, norm=norm, aspect="auto")
    # a surface gap between the task columns and the mean column
    ax.axvline(len(tasks) - 0.5, color=SURFACE, linewidth=4)
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            fill = DIVERGING(norm(table[i, j]))
            ax.text(j, i, signed(table[i, j]), ha="center", va="center", fontsize=8, color=ink_on(fill),
                    fontweight="bold" if j == len(tasks) else "normal")
    ax.set_xticks(range(len(tasks) + 1), tasks + ["mean"])
    ax.set_yticks(range(len(methods)), [label(m) for m in methods])
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    bar = fig.colorbar(image, ax=ax, fraction=0.04, pad=0.02)
    bar.outline.set_visible(False)
    bar.ax.tick_params(length=0)
    bar.set_label("forward transfer (0: as scratch)")
    ax.set_title("Forward transfer of training success rate vs scratch", loc="left")
    fig.tight_layout()
    save(fig, output_dir, "forward_transfer")


def plot_transfer_matrices(matrices: dict, tasks: list[str], output_dir: str):
    methods = [m for m in METHODS if m in matrices]
    columns = 4
    rows = int(np.ceil(len(methods) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(3.0 * columns, 2.9 * rows + 0.6), squeeze=False)
    vmax = max(matrix["error"].max() for matrix in matrices.values())
    norm = Normalize(vmin=0.0, vmax=vmax)
    short = [task.replace("vertical_circle", "v_circle") for task in tasks]
    for index, ax in enumerate(axes.flat):
        if index >= len(methods):
            ax.set_visible(False)
            continue
        method = methods[index]
        error, crash = matrices[method]["error"], matrices[method]["crash"]
        image = ax.imshow(error, cmap=BLUES, norm=norm, aspect="equal")
        for i in range(error.shape[0]):
            for j in range(error.shape[1]):
                if crash[i, j] >= 0.5:
                    # tone-on-tone: dark hatch on light cells, light hatch on dark cells
                    light = ink_on(BLUES(norm(error[i, j]))) == INK
                    hatch_ink = HATCH_INK if light else (1.0, 1.0, 1.0, 0.55)
                    ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, hatch="////",
                                           edgecolor=hatch_ink, linewidth=0, zorder=3))
        # the diagonal: each task right after training on it; inset so the cell gap does not cover it
        for i in range(min(error.shape)):
            ax.add_patch(Rectangle((i - 0.36, i - 0.36), 0.72, 0.72, fill=False,
                                   edgecolor=ink_on(BLUES(norm(error[i, i]))), linewidth=1.2, zorder=4))
        ax.set_title(label(method), loc="left")
        ax.set_xticks(range(len(tasks)), short, rotation=45, ha="right", fontsize=7.5)
        ax.set_yticks(range(len(tasks)), short if index % columns == 0 else [], fontsize=7.5)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        # a surface gap between cells (interior boundaries only)
        for boundary in np.arange(0.5, len(tasks) - 1):
            ax.axvline(boundary, color=SURFACE, linewidth=2, zorder=2)
            ax.axhline(boundary, color=SURFACE, linewidth=2, zorder=2)
        ax.set_xlim(-0.5, len(tasks) - 0.5)
        ax.set_ylim(len(tasks) - 0.5, -0.5)
    fig.supxlabel("evaluated on task", color=INK_SECONDARY, fontsize=9)
    fig.supylabel("policy after training on task", color=INK_SECONDARY, fontsize=9)
    bar = fig.colorbar(image, ax=axes, fraction=0.02, pad=0.02)
    bar.outline.set_visible(False)
    bar.ax.tick_params(length=0)
    bar.set_label("mean position error [cm]")
    fig.legend(handles=[
        Patch(facecolor="#ffffff", edgecolor=HATCH_INK, hatch="////", label="crash rate ≥ 50%"),
        Patch(facecolor="none", edgecolor=INK, label="task just trained (diagonal, inner box)"),
    ], loc="upper right", ncol=2, bbox_to_anchor=(0.9, 1.0))
    fig.suptitle("Transfer matrices: error of each stage's policy on every task", x=0.01, ha="left", fontsize=11)
    save(fig, output_dir, "transfer_matrices")


def tag_curve(tag: str, scale: float = 1.0):
    """Reader of one TensorBoard scalar of a stage directory: (steps, values * scale)."""
    def read(directory: str):
        values = scalars(directory, tag)
        steps = np.array(sorted(values))
        return steps, np.array([values[s] for s in steps]) * scale
    return read


def success_percent(directory: str):
    steps, success = success_curve(directory)
    return steps, success * 100.0


def plot_curve_grid(metrics: dict, tasks: list[str], rows: list[tuple], title: str, output_dir: str, name: str):
    """One column per task, one row per (y label, reader, y limits or None, share y across tasks)."""
    methods = [m for m in METHODS if m in metrics["methods"]]
    fig, axes = plt.subplots(len(rows), len(tasks), figsize=(3.0 * len(tasks), 2.7 * len(rows) + 0.6),
                             sharex=True, squeeze=False)
    curves = {(r, task): [] for r in range(len(rows)) for task in tasks}
    for method in methods:
        with open(os.path.join(metrics["methods"][method]["session"], "session.json")) as f:
            stages = {stage["task"]: stage for stage in json.load(f)["stages"]}
        style = dict(color=color(method), marker=marker(method), markersize=5, markevery=12,
                     markeredgecolor=SURFACE, markeredgewidth=0.8, label=label(method))
        for column, task in enumerate(tasks):
            if task not in stages:
                continue
            directory = stage_dir(stages[task])
            for r, (_, read, _, _) in enumerate(rows):
                steps, values = read(directory)
                curves[(r, task)].append(values)
                axes[r, column].plot(steps, values, **style)
    for r, (ylabel, _, limits, share) in enumerate(rows):
        if share:
            for ax in axes[r, 1:]:
                ax.sharey(axes[r, 0])
                ax.tick_params(labelleft=False)
        if limits is None and share:
            # axes were linked after plotting: set the range from the data of all tasks
            values = np.concatenate([c for task in tasks for c in curves[(r, task)]])
            low, high = np.nanmin(values), np.nanmax(values)
            margin = 0.05 * (high - low) if high > low else 1.0
            limits = (low - margin, high + margin)
        if limits is not None:
            axes[r, 0].set_ylim(*limits)
        axes[r, 0].set_ylabel(ylabel)
        for column in range(len(tasks)):
            axes[r, column].grid(axis="y")
    for column, task in enumerate(tasks):
        groups = [curves[(r, task)] for r in range(len(rows))]
        if all(len(g) > 1 and all(len(c) == len(g[0]) and np.allclose(c, g[0], rtol=0.02, atol=1e-6) for c in g)
               for g in groups):
            # the last method drawn covers the others; one note for the whole column
            axes[0, column].text(0.03, 0.55, "all methods overlap", transform=axes[0, column].transAxes,
                                 color=INK_SECONDARY, fontsize=8)
    for column, task in enumerate(tasks):
        axes[0, column].set_title(task, loc="left")
        axes[-1, column].set_xlabel("training steps")
    handles = [Line2D([], [], color=color(m), marker=marker(m), markersize=6, markeredgecolor=SURFACE,
                      label=label(m)) for m in methods]
    fig.legend(handles=handles, loc="upper center", ncol=len(methods), bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(title, x=0.01, y=1.05, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, output_dir, name)


def plot_learning_curves(metrics: dict, tasks: list[str], output_dir: str):
    rows = [("success rate [%]", success_percent, (-3, 103), True),
            ("position error [cm]", tag_curve("Info / Metrics/position_error", 100.0), None, True)]
    plot_curve_grid(metrics, tasks, rows, "Training curves (stochastic policy, training episodes)", output_dir,
                    "learning_curves")


def plot_reward_curves(metrics: dict, tasks: list[str], output_dir: str):
    # episode return grows with episode length, which differs per task: y is not shared across tasks
    rows = [("episode return (mean)", tag_curve("Reward / Total reward (mean)"), None, False),
            ("reward per step (mean)", tag_curve("Reward / Instantaneous reward (mean)"), None, True)]
    plot_curve_grid(metrics, tasks, rows, "Reward during training (stochastic policy, training episodes)",
                    output_dir, "reward_curves")


def plot_cost(metrics: dict, tasks: list[str], output_dir: str):
    series = {}
    for method, result in metrics["methods"].items():
        series[method] = result["cost"]["policy"]["stages"]
        if "meta_policy" in result["cost"]:
            series["fame_meta"] = result["cost"]["meta_policy"]["stages"]
    panels = [("parameters", "Parameters", 1.0), ("flops_per_action", "kFLOPs per action", 1e-3)]
    if all("cpu_latency_us" in stage for stages in series.values() for stage in stages):
        panels.append(("cpu_latency_us", "CPU latency per action [µs]", 1.0))
    # methods whose policy does not grow draw the same line: draw it once, in the reference gray
    constant = [m for m in METHODS if m in series and len({s["parameters"] for s in series[m]}) == 1]
    growing = [m for m in METHODS if m in series and m not in constant]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 3.4))
    x = np.arange(len(tasks))
    for ax, (key, title, scale) in zip(np.atleast_1d(axes), panels):
        lines = [(m, label(m), color(m), marker(m)) for m in growing]
        if constant:
            lines.insert(0, (constant[0], "MLP: " + ", ".join(label(m) for m in constant), MUTED, "o"))
        for method, name, line_color, line_marker in lines:
            values = [stage[key] * scale for stage in series[method]]
            ax.plot(x[: len(values)], values, color=line_color, marker=line_marker, markersize=6,
                    markeredgecolor=SURFACE, markeredgewidth=0.8, label=name)
        ax.set_ylim(bottom=0)
        ax.yaxis.set_major_formatter(THOUSANDS)
        ax.set_title(title, loc="left")
        ax.set_xticks(x, [task.replace("vertical_circle", "v_circle") for task in tasks], rotation=30, ha="right")
        ax.grid(axis="y", which="major")
    np.atleast_1d(axes)[0].set_xlabel("policy after training on task")
    handles, names = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.suptitle("Cost of the policy after each stage", x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.legend(handles, names, loc="lower center", ncol=len(names), bbox_to_anchor=(0.5, 0.0))
    save(fig, output_dir, "cost")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--metrics", required=True, help="JSON written by scripts/skrl/continual_metrics.py.")
    parser.add_argument("--output_dir", required=True, help="Directory for the figures.")
    args = parser.parse_args()
    with open(args.metrics) as f:
        metrics = json.load(f)
    os.makedirs(args.output_dir, exist_ok=True)

    tasks = next(iter(metrics["methods"].values()))["tasks"]
    unknown = [m for m in metrics["methods"] if m not in METHODS]
    if unknown:
        raise ValueError(f"No style for methods {unknown}; add them to METHODS")
    matrices = transfer_matrices(metrics, tasks)
    matrices = {m: matrices[m] for m in METHODS if m in matrices}
    values = summary_values(metrics, matrices)

    plot_summary(values, args.output_dir)
    plot_forward_transfer(metrics, tasks, args.output_dir)
    plot_transfer_matrices(matrices, tasks, args.output_dir)
    plot_learning_curves(metrics, tasks, args.output_dir)
    plot_reward_curves(metrics, tasks, args.output_dir)
    plot_cost(metrics, tasks, args.output_dir)


if __name__ == "__main__":
    main()
