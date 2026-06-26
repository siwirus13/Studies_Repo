"""
plot_learning_curve.py
======================
Reads SB3 TensorBoard event files from ./logs/ and produces a two-panel
learning-curve figure suitable for the LaTeX report.

Panels:
  Top    — Episode reward (mean ± std band, smoothed)
  Bottom — Eval reward from EvalCallback (deterministic, every 25k steps)

Phase boundaries are drawn as vertical lines with shaded region labels.

Usage:
    python plot_learning_curve.py                          # default ./logs
    python plot_learning_curve.py --logdir /path/to/logs
    python plot_learning_curve.py --out fig_learning.pdf  # vector output
    python plot_learning_curve.py --smooth 0.97           # heavier smoothing
"""

import argparse
import os
import glob
import sys
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# ─── Curriculum definition (must match train_ppo.py) ─────────────────────────
PHASES = [
    {"name": "Phase 1\nCalm air",          "steps": 2_000_000, "wind": "0 m/s"},
    {"name": "Phase 2\nSteady wind",        "steps": 5_000_000, "wind": "5 m/s"},
    {"name": "Phase 3\nWind + turbulence",  "steps": 4_000_000, "wind": "10 m/s"},
    {"name": "Phase 4\nFull chaos",         "steps": 2_000_000, "wind": "15 m/s"},
]

# Cumulative step boundaries
_cumsteps = [0]
for p in PHASES:
    _cumsteps.append(_cumsteps[-1] + p["steps"])
PHASE_BOUNDS = _cumsteps          # [0, 2M, 7M, 11M, 13M]

PHASE_COLORS = ["#d0e8ff", "#c8f0d8", "#fff0c0", "#ffd5c8"]

# ─── Scalar tags written by SB3 ──────────────────────────────────────────────
TAG_REWARD    = "rollout/ep_rew_mean"
TAG_EPLEN     = "rollout/ep_len_mean"
TAG_EVAL      = "eval/mean_reward"
TAG_EVAL_LEN  = "eval/mean_ep_length"


# ─── Helpers ─────────────────────────────────────────────────────────────────

def ema_smooth(values: np.ndarray, alpha: float) -> np.ndarray:
    """Exponential moving average smoothing."""
    out = np.empty_like(values, dtype=np.float64)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = alpha * out[i - 1] + (1 - alpha) * values[i]
    return out


def _parse_tensor_value(tensor_proto) -> float:
    """Parse a TF2 tensor_proto scalar to a Python float."""
    try:
        import struct
        raw = tensor_proto.tensor_content
        if raw:
            return struct.unpack("f", raw[:4])[0]
        if tensor_proto.float_val:
            return float(tensor_proto.float_val[0])
        if tensor_proto.double_val:
            return float(tensor_proto.double_val[0])
    except Exception:
        pass
    return float("nan")


def load_scalars(logdir: str, tag: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Walk *logdir* recursively, collect all (step, value) pairs for *tag*,
    sort by step, and return (steps, values) as numpy arrays.

    Handles both formats:
      - TF1 / SB3-legacy : stored as 'scalars' in the event accumulator
      - TF2 / SB3-current : stored as 'tensors' (tf.summary.scalar writes tensors in TF2)
    """
    event_files = glob.glob(os.path.join(logdir, "**", "events.out.tfevents.*"),
                            recursive=True)
    if not event_files:
        event_files = glob.glob(os.path.join(logdir, "events.out.tfevents.*"))

    records: list[tuple[int, float]] = []

    for ef in event_files:
        ea = EventAccumulator(ef, size_guidance={"scalars": 0, "tensors": 0})
        ea.Reload()
        tags = ea.Tags()

        if tag in tags.get("scalars", []):
            # TF1-style: direct scalar value
            for e in ea.Scalars(tag):
                records.append((e.step, e.value))

        elif tag in tags.get("tensors", []):
            # TF2-style: scalar wrapped in a tensor proto
            for e in ea.Tensors(tag):
                val = _parse_tensor_value(e.tensor_proto)
                if not (val != val):   # skip NaN
                    records.append((e.step, val))

    if not records:
        return np.array([]), np.array([])

    records.sort(key=lambda x: x[0])
    steps  = np.array([r[0] for r in records], dtype=np.float64)
    values = np.array([r[1] for r in records], dtype=np.float64)

    # De-duplicate steps (keep first per step; can happen on resume)
    _, unique_idx = np.unique(steps, return_index=True)
    return steps[unique_idx], values[unique_idx]


def draw_phase_bands(ax, total_steps: float):
    """Shade phase regions and add vertical boundary lines + labels."""
    for i, phase in enumerate(PHASES):
        x0 = PHASE_BOUNDS[i]
        x1 = PHASE_BOUNDS[i + 1]
        if x0 >= total_steps:
            break
        x1 = min(x1, total_steps)
        ax.axvspan(x0, x1, color=PHASE_COLORS[i], alpha=0.30, zorder=0)
        mid = (x0 + x1) / 2
        # Short label in millions
        ax.text(mid, 1.01, phase["name"].split("\n")[0],
                transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=7.5,
                color="#555555", style="italic")

    # Boundary lines (skip 0 and last)
    for b in PHASE_BOUNDS[1:-1]:
        if b < total_steps:
            ax.axvline(b, color="#888888", lw=0.9, ls="--", zorder=1)


def millions_formatter(x, _pos):
    return f"{x/1e6:.0f}M" if x >= 1e6 else f"{x/1e3:.0f}k"


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Plot PPO rocket learning curves")
    parser.add_argument("--logdir",  default="./logs",
                        help="Root TensorBoard log directory (default: ./logs)")
    parser.add_argument("--out",     default="fig_learning_curve.pdf",
                        help="Output file (.pdf / .png / .svg)")
    parser.add_argument("--smooth",  type=float, default=0.93,
                        help="EMA smoothing factor 0‒1 (higher = smoother, default 0.93)")
    parser.add_argument("--dpi",     type=int,   default=200,
                        help="DPI for raster output (default 200)")
    args = parser.parse_args()

    if not os.path.isdir(args.logdir):
        sys.exit(f"ERROR: Log directory not found: {args.logdir!r}\n"
                 "       Start training first, or pass --logdir <path>.")

    print(f"Scanning logs in: {os.path.abspath(args.logdir)}")

    # ── Load data ──────────────────────────────────────────────────────────
    rew_steps,  rew_vals  = load_scalars(args.logdir, TAG_REWARD)
    eval_steps, eval_vals = load_scalars(args.logdir, TAG_EVAL)
    len_steps,  len_vals  = load_scalars(args.logdir, TAG_EPLEN)

    if len(rew_steps) == 0:
        sys.exit(
            f"ERROR: No '{TAG_REWARD}' scalars found under {args.logdir!r}.\n"
            "       Make sure training has been run and logs exist."
        )

    total_steps = max(
        rew_steps.max() if len(rew_steps) else 0,
        eval_steps.max() if len(eval_steps) else 0,
    )
    print(f"  rollout/ep_rew_mean : {len(rew_steps):,} points  (up to {total_steps/1e6:.2f} M steps)")
    print(f"  eval/mean_reward    : {len(eval_steps):,} points")

    has_eval = len(eval_steps) > 0
    n_panels  = 2 if has_eval else 1

    # ── Figure layout ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(
        n_panels, 1,
        figsize=(7.5, 4.5 if has_eval else 3.2),
        sharex=True,
        gridspec_kw={"hspace": 0.08, "height_ratios": [2, 1] if has_eval else [1]},
    )
    if n_panels == 1:
        axes = [axes]

    ax_rew  = axes[0]
    ax_eval = axes[1] if has_eval else None

    # ── Top panel: episode reward ───────────────────────────────────────────
    draw_phase_bands(ax_rew, total_steps)

    # Raw (faint)
    ax_rew.plot(rew_steps, rew_vals, color="#3a86ff", alpha=0.18, lw=0.7, zorder=2)

    # Smoothed
    smoothed = ema_smooth(rew_vals, args.smooth)
    ax_rew.plot(rew_steps, smoothed, color="#3a86ff", lw=1.8, zorder=3,
                label="Mean episode reward (smoothed)")

    ax_rew.set_ylabel("Episode reward", fontsize=10)
    ax_rew.legend(loc="upper left", fontsize=8, framealpha=0.7)
    ax_rew.grid(axis="y", ls=":", lw=0.6, alpha=0.6)
    ax_rew.set_xlim(0, total_steps * 1.01)
    ax_rew.spines[["top", "right"]].set_visible(False)
    ax_rew.tick_params(labelbottom=False)

    # Annotate max reward
    best_idx = np.argmax(smoothed)
    ax_rew.annotate(
        f"peak {smoothed[best_idx]:.1f}",
        xy=(rew_steps[best_idx], smoothed[best_idx]),
        xytext=(rew_steps[best_idx] + total_steps * 0.03, smoothed[best_idx] * 0.95),
        fontsize=7.5, color="#1a5fbf",
        arrowprops=dict(arrowstyle="-|>", color="#1a5fbf", lw=0.8),
    )

    # ── Bottom panel: eval reward ───────────────────────────────────────────
    if ax_eval is not None:
        draw_phase_bands(ax_eval, total_steps)
        ax_eval.plot(eval_steps, eval_vals, color="#ff6b35",
                     marker="o", ms=3, lw=1.4, zorder=3,
                     label="Eval reward (det., 5 eps)")
        ax_eval.set_ylabel("Eval reward", fontsize=10)
        ax_eval.legend(loc="upper left", fontsize=8, framealpha=0.7)
        ax_eval.grid(axis="y", ls=":", lw=0.6, alpha=0.6)
        ax_eval.spines[["top", "right"]].set_visible(False)

    # ── Shared x-axis ──────────────────────────────────────────────────────
    bottom_ax = axes[-1]
    bottom_ax.set_xlabel("Training steps", fontsize=10)
    bottom_ax.xaxis.set_major_formatter(plt.FuncFormatter(millions_formatter))

    # ── Phase legend patches ────────────────────────────────────────────────
    legend_patches = [
        mpatches.Patch(color=PHASE_COLORS[i], alpha=0.6,
                       label=f"Ph.{i+1}: {PHASES[i]['wind']}")
        for i in range(len(PHASES))
    ]
    fig.legend(handles=legend_patches,
               loc="lower center", ncol=4,
               fontsize=7.5, framealpha=0.7,
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("PPO Rocket Attitude Control — Training Curve", fontsize=12, y=1.01)

    # ── Save ───────────────────────────────────────────────────────────────
    out_path = args.out
    plt.tight_layout()
    plt.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    print(f"\nSaved: {os.path.abspath(out_path)}")
    print(f"Include in LaTeX with:\n  \\includegraphics[width=\\linewidth]{{{out_path}}}")


if __name__ == "__main__":
    main()
