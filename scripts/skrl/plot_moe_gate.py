"""Plot the KAE-MoE gate value (TensorBoard tag ``Policy / MoE gate (mean)``) of every stage of a session.

The gate ``g`` in [0, 1] blends the two pathways: ``action = g * kae_action + (1 - g) * mlp_action``.
Stage 0 trains a plain MLP policy, so it has no gate.

Usage:
    python scripts/skrl/plot_moe_gate.py <session_dir> --out docs/figures/<name>/moe_gate
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from continual_metrics import scalars, stage_dir  # noqa: E402

TAG = "Policy / MoE gate (mean)"
COLOR = "#2a78d6"
MUTED = "#52514e"
GRID = "#e4e3df"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("session", help="KAE-MoE session directory (contains session.json)")
    parser.add_argument("--out", required=True, help="output path without extension (.png and .pdf are written)")
    args = parser.parse_args()

    with open(os.path.join(args.session, "session.json")) as f:
        session = json.load(f)
    stages = session["stages"]

    fig, axes = plt.subplots(1, len(stages), figsize=(3.0 * len(stages), 2.8), sharey=True)
    for index, (ax, stage) in enumerate(zip(axes, stages)):
        values = scalars(stage_dir(stage), TAG)
        ax.set_title(f"Stage {index}: {stage['task']}", fontsize=10)
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.set_xlabel("Timestep", fontsize=9, color=MUTED)
        if not values:
            ax.text(0.5, 0.5, "Plain MLP policy\n(no gate)", transform=ax.transAxes,
                    ha="center", va="center", fontsize=9, color=MUTED)
            ax.set_xticks([])
            continue
        steps = sorted(values)
        ax.plot(steps, [values[s] for s in steps], color=COLOR, linewidth=2)
        best = stage.get("best_timestep")
        if best is not None:
            ax.axvline(best, color=MUTED, linestyle="--", linewidth=1)
            # put the label left of the line when the line is near the right end of the axis
            right = best > 0.75 * steps[-1]
            ax.annotate("selected\ncheckpoint", xy=(best, 0.04), xytext=(-4 if right else 4, 0),
                        textcoords="offset points", fontsize=7, color=MUTED, va="bottom",
                        ha="right" if right else "left")
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x / 1000:g}k"))
    axes[0].set_ylabel("Gate g (mean)", fontsize=9, color=MUTED)
    fig.suptitle("KAE-MoE gate during training  (g = 1: KAE pathway only, g = 0: MLP pathway only)", fontsize=10)
    fig.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{args.out}.{ext}", dpi=200)
    print(f"[INFO] Wrote {args.out}.png and {args.out}.pdf")


if __name__ == "__main__":
    main()
