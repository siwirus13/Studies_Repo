"""
visualize_3d.py
===============
Run one episode with a trained PPO model and show two things:

  1. Live terminal dashboard — prints a real-time observation table
     each step so you can watch what the model "sees" as it flies.

  2. 3D trajectory plot — after the episode ends, renders the full
     flight path in 3D (matplotlib) with:
       - Path coloured by tilt angle (green → red)
       - Nose-direction arrows sampled every N steps
       - Side panels showing altitude vs time and tilt vs time
       - Summary stats in the title

Usage:
    python visualize_3d.py --model ./checkpoints/best_model.zip
    python visualize_3d.py --model ./checkpoints/best_model.zip --save flight3d.png
    python visualize_3d.py --model ./checkpoints/best_model.zip --no-live  # skip terminal output
    python visualize_3d.py --model ./checkpoints/best_model.zip --wind 0   # calm air

Requires:
    pip install stable-baselines3 matplotlib numpy
"""

import argparse
import math
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401 (registers 3d projection)
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from stable_baselines3 import PPO

from rocket_env_jsbsim import RocketAeroJSBSimEnv


# ─────────────────────────────────────────────────────────────────────────────
# Colour helpers
# ─────────────────────────────────────────────────────────────────────────────

def _tilt_colour(tilt_deg: float, max_tilt: float = 30.0):
    """Map tilt angle to an RGB tuple: green (0°) → amber → red (max_tilt°)."""
    t = min(1.0, tilt_deg / max_tilt)
    if t < 0.5:
        r, g, b = t * 2, 1.0, 0.0
    else:
        r, g, b = 1.0, 1.0 - (t - 0.5) * 2, 0.0
    return (r, g, b)


def _make_segments(xs, ys, zs):
    """Build list of (2,3) segments for Line3DCollection from coordinate arrays."""
    pts = np.column_stack([xs, ys, zs])
    return [pts[i : i + 2] for i in range(len(pts) - 1)]


# ─────────────────────────────────────────────────────────────────────────────
# Live terminal dashboard
# ─────────────────────────────────────────────────────────────────────────────

HEADER = (
    "  {:>7s}  {:>8s}  {:>7s}  {:>7s}  {:>8s}  {:>8s}  {:>8s}  {:>8s}  {:>8s}"
).format("t(s)", "alt(m)", "tilt(°)", "v_up", "tilt_x", "tilt_y", "p", "q", "r")

DIVIDER = "  " + "─" * (len(HEADER) - 2)

ROW_FMT = (
    "  {:>7.2f}  {:>8.1f}  {:>7.2f}  {:>7.1f}  {:>+8.4f}  {:>+8.4f}"
    "  {:>+8.3f}  {:>+8.3f}  {:>+8.3f}"
)


def _live_header():
    print()
    print(DIVIDER)
    print(HEADER)
    print(DIVIDER)


def _live_row(t, obs, info):
    tilt_x, tilt_y = obs[0], obs[1]
    p, q, r        = obs[3], obs[4], obs[5]
    h_km, v_up     = obs[6], obs[7]
    tilt_deg       = info["tilt_deg"]
    print(ROW_FMT.format(t, h_km * 1000, tilt_deg, v_up, tilt_x, tilt_y, p, q, r))


# ─────────────────────────────────────────────────────────────────────────────
# Episode runner
# ─────────────────────────────────────────────────────────────────────────────

def run_episode(model, env, deterministic=True, live=True, live_every=25):
    """Run one episode and return a flight log dict."""
    obs, _ = env.reset()

    log = dict(
        t=[], alt_m=[], tilt_deg=[], tilt_x=[], tilt_y=[],
        p=[], q=[], r=[], v_up=[],
        a_elev=[], a_rud=[], a_ail=[],
        thrust_n=[],
        # horizontal dead-reckoning position [m]
        north=[], east=[],
    )
    north, east = 0.0, 0.0

    if live:
        _live_header()

    step = 0
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, _reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        t       = info["sim_time"]
        tilt_x  = float(obs[0])
        tilt_y  = float(obs[1])
        h_km    = float(obs[6])
        v_up    = float(obs[7])
        dt      = env.dt

        # Dead-reckon horizontal drift: velocity component ≈ v_up * sin(tilt)
        # tilt_x/tilt_y are already the horizontal components of the nose unit vector
        # so horizontal speed ≈ v_up * tilt (accurate for small tilts)
        north += v_up * tilt_x * dt
        east  += v_up * tilt_y * dt

        log["t"].append(t)
        log["alt_m"].append(h_km * 1000)
        log["tilt_deg"].append(info["tilt_deg"])
        log["tilt_x"].append(tilt_x)
        log["tilt_y"].append(tilt_y)
        log["p"].append(math.degrees(float(obs[3])))
        log["q"].append(math.degrees(float(obs[4])))
        log["r"].append(math.degrees(float(obs[5])))
        log["v_up"].append(v_up)
        log["a_elev"].append(float(action[0]))
        log["a_rud"].append(float(action[1]))
        log["a_ail"].append(float(action[2]))
        log["thrust_n"].append(info["thrust_n"])
        log["north"].append(north)
        log["east"].append(east)

        if live and step % live_every == 0:
            _live_row(t, obs, info)

        step += 1

    if live:
        print(DIVIDER)

    outcome = "✓ REACHED 10 km" if truncated else "✗ TERMINATED EARLY"
    max_tilt = max(log["tilt_deg"])
    max_alt  = max(log["alt_m"])

    print(f"\n  Outcome   : {outcome}")
    print(f"  Duration  : {log['t'][-1]:.2f} s  ({step} steps)")
    print(f"  Max alt   : {max_alt:.1f} m")
    print(f"  Max tilt  : {max_tilt:.2f}°")
    print(f"  Horiz drift: N={north:.1f} m  E={east:.1f} m")
    print()

    return log, truncated


# ─────────────────────────────────────────────────────────────────────────────
# 3-D plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_3d(log, save_path=None, arrow_every=50):
    t       = np.array(log["t"])
    alt     = np.array(log["alt_m"])
    north   = np.array(log["north"])
    east    = np.array(log["east"])
    tilt    = np.array(log["tilt_deg"])
    tilt_x  = np.array(log["tilt_x"])
    tilt_y  = np.array(log["tilt_y"])
    v_up    = np.array(log["v_up"])

    max_tilt = max(tilt.max(), 1.0)

    fig = plt.figure(figsize=(14, 9))
    fig.patch.set_facecolor("#0e0e12")

    gs = gridspec.GridSpec(
        2, 2,
        figure=fig,
        left=0.04, right=0.98,
        top=0.91,  bottom=0.06,
        wspace=0.28, hspace=0.38,
    )

    ax3d   = fig.add_subplot(gs[:, 0], projection="3d")
    ax_alt = fig.add_subplot(gs[0, 1])
    ax_tlt = fig.add_subplot(gs[1, 1])

    for ax in [ax_alt, ax_tlt]:
        ax.set_facecolor("#14141a")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333344")
        ax.tick_params(colors="#aaaacc", labelsize=8)
        ax.xaxis.label.set_color("#aaaacc")
        ax.yaxis.label.set_color("#aaaacc")
        ax.grid(alpha=0.15, color="#444466")

    ax3d.set_facecolor("#0e0e12")
    ax3d.xaxis.pane.fill = ax3d.yaxis.pane.fill = ax3d.zaxis.pane.fill = False
    ax3d.xaxis.pane.set_edgecolor("#1e1e2e")
    ax3d.yaxis.pane.set_edgecolor("#1e1e2e")
    ax3d.zaxis.pane.set_edgecolor("#1e1e2e")
    ax3d.tick_params(colors="#aaaacc", labelsize=7)
    ax3d.xaxis.label.set_color("#aaaacc")
    ax3d.yaxis.label.set_color("#aaaacc")
    ax3d.zaxis.label.set_color("#aaaacc")
    ax3d.grid(True, alpha=0.1)

    # ── 3D path coloured by tilt ──────────────────────────────────────────
    segment_colours = [
        _tilt_colour(0.5 * (tilt[i] + tilt[i + 1]), max_tilt)
        for i in range(len(tilt) - 1)
    ]
    segs = _make_segments(east, north, alt)   # X=east, Y=north, Z=alt
    lc = Line3DCollection(segs, colors=segment_colours, linewidths=1.6, alpha=0.9)
    ax3d.add_collection3d(lc)

    # ── Nose-direction arrows ─────────────────────────────────────────────
    # Nose unit vector in NED, projected: tilt_x=north, tilt_y=east, up=sqrt(1-tx²-ty²)
    arrow_len = max(alt.max() * 0.06, 200.0)
    for i in range(0, len(t), arrow_every):
        tx, ty = tilt_x[i], tilt_y[i]
        tz_up  = math.sqrt(max(0.0, 1.0 - tx**2 - ty**2))
        ax3d.quiver(
            east[i], north[i], alt[i],
            ty * arrow_len, tx * arrow_len, tz_up * arrow_len,
            color="#5588ff", linewidth=0.8, arrow_length_ratio=0.25, alpha=0.7,
        )

    # ── Launch and current markers ────────────────────────────────────────
    ax3d.scatter([east[0]],  [north[0]],  [alt[0]],  s=60,  color="#22dd88", zorder=5, label="Launch")
    ax3d.scatter([east[-1]], [north[-1]], [alt[-1]], s=80,  color="#ff4455", zorder=5, label="End")

    # ── Ground shadow ─────────────────────────────────────────────────────
    ax3d.plot(east, north, zs=0, zdir="z", color="#334433", linewidth=0.7, alpha=0.5, linestyle="--")

    ax3d.set_xlabel("East (m)",   labelpad=4)
    ax3d.set_ylabel("North (m)",  labelpad=4)
    ax3d.set_zlabel("Alt (m)",    labelpad=4)
    ax3d.legend(fontsize=7, facecolor="#1a1a24", edgecolor="#333344", labelcolor="#ccccdd", loc="upper left")

    # ── Colour bar (tilt legend) ──────────────────────────────────────────
    sm = plt.cm.ScalarMappable(
        cmap=matplotlib.colors.LinearSegmentedColormap.from_list(
            "tilt", [(0, "#00cc66"), (0.5, "#ffaa00"), (1, "#ff2233")]
        ),
        norm=matplotlib.colors.Normalize(vmin=0, vmax=max_tilt),
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax3d, shrink=0.55, pad=0.08, aspect=18)
    cbar.set_label("Tilt angle (°)", color="#aaaacc", fontsize=8)
    cbar.ax.yaxis.set_tick_params(color="#aaaacc", labelsize=7)
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="#aaaacc")

    # ── Side panel: altitude vs time ──────────────────────────────────────
    ax_alt.plot(t, alt / 1000, color="#4499ff", linewidth=1.4)
    ax_alt.axhline(25.0, color="#22dd88", linewidth=0.8, linestyle="--", alpha=0.7, label="Target 25 km")
    ax_alt.fill_between(t, alt / 1000, alpha=0.12, color="#4499ff")
    ax_alt.set_xlabel("Time (s)", fontsize=8)
    ax_alt.set_ylabel("Altitude (km)", fontsize=8)
    ax_alt.set_title("Altitude", color="#ccccdd", fontsize=9, pad=4)
    ax_alt.legend(fontsize=7, facecolor="#14141a", edgecolor="#333344", labelcolor="#ccccdd")

    # ── Side panel: tilt vs time with danger zone ─────────────────────────
    ax_tlt.plot(t, tilt, color="#ffaa33", linewidth=1.4, label="Tilt°")
    ax_tlt.axhline(30.0, color="#ff4455", linewidth=0.8, linestyle="--", alpha=0.8, label="Term. limit 30°")
    ax_tlt.fill_between(t, tilt, alpha=0.15, color="#ffaa33")
    ax_tlt.set_xlabel("Time (s)", fontsize=8)
    ax_tlt.set_ylabel("Tilt (°)", fontsize=8)
    ax_tlt.set_title("Tilt angle", color="#ccccdd", fontsize=9, pad=4)
    ax_tlt.legend(fontsize=7, facecolor="#14141a", edgecolor="#333344", labelcolor="#ccccdd")

    # ── Title ─────────────────────────────────────────────────────────────
    outcome_str = "✓ reached 25 km" if alt[-1] >= 24_800 else f"✗ ended at {alt[-1]:.0f} m"
    fig.suptitle(
        f"Rocket flight  ·  {outcome_str}  ·  {t[-1]:.1f} s  ·  "
        f"max tilt {tilt.max():.1f}°  ·  drift N={north[-1]:.0f} m  E={east[-1]:.0f} m",
        color="#ddddee", fontsize=10, y=0.97,
    )

    if save_path:
        plt.savefig(save_path, dpi=150, facecolor=fig.get_facecolor())
        print(f"  Saved 3D plot → {save_path}")
    else:
        plt.show()

    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run a trained PPO rocket model and show a 3D flight trajectory."
    )
    parser.add_argument("--model",       type=str, required=True,
                        help="Path to .zip model file, e.g. checkpoints/best_model.zip")
    parser.add_argument("--save",        type=str, default=None,
                        help="Save the 3D plot to this file instead of displaying it")
    parser.add_argument("--wind",        type=float, default=10.0,
                        help="Max steady crosswind [m/s] (default 10)")
    parser.add_argument("--no-turb",     action="store_true",
                        help="Disable Dryden turbulence")
    parser.add_argument("--stochastic",  action="store_true",
                        help="Sample actions stochastically instead of greedy")
    parser.add_argument("--no-live",     action="store_true",
                        help="Skip the real-time terminal observation printout")
    parser.add_argument("--live-every",  type=int, default=25,
                        help="Print a live row every N steps (default 25 = every 0.5 s)")
    parser.add_argument("--arrow-every", type=int, default=50,
                        help="Draw a nose-direction arrow every N steps in 3D plot")
    parser.add_argument("--seed",        type=int, default=None,
                        help="Random seed for env reset")
    args = parser.parse_args()

    print(f"\n  Loading model : {args.model}")
    model = PPO.load(args.model)

    env = RocketAeroJSBSimEnv(
        wind_speed_mps  = args.wind,
        wind_turbulence = not args.no_turb,
    )
    if args.seed is not None:
        env.reset(seed=args.seed)

    print(f"  Wind          : {args.wind} m/s{'  (no turbulence)' if args.no_turb else ' + Dryden turbulence'}")
    print(f"  Policy        : {'stochastic' if args.stochastic else 'deterministic'}")
    print(f"  Live output   : {'off' if args.no_live else f'every {args.live_every} steps'}")
    print()

    log, truncated = run_episode(
        model, env,
        deterministic = not args.stochastic,
        live          = not args.no_live,
        live_every    = args.live_every,
    )

    env.close()

    print("  Rendering 3D plot…")
    plot_3d(log, save_path=args.save, arrow_every=args.arrow_every)


if __name__ == "__main__":
    main()
