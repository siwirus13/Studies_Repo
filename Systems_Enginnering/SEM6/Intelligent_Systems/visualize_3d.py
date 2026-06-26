"""
visualize.py
===============
Run one episode with a trained PPO model and generate two visualizations:

  1. 3D trajectory plot — renders the full flight path in 3D (matplotlib)
  2. 2D telemetry plot — graphs attitude, rates, altitude, thrust, and controls

Usage:
    python visualize.py --model ./checkpoints/best_model.zip --phase 2
    python visualize.py --model ./checkpoints/best_model.zip --phase 4 --save-3d flight3d.png --save-2d telemetry.png
    python visualize.py --model ./checkpoints/best_model.zip --no-live  # skip terminal output

Requires:
    pip install stable-baselines3 matplotlib numpy
"""

import argparse
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401 (registers 3d projection)
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from stable_baselines3 import PPO

from rocket_env_jsbsim import RocketAeroJSBSimEnv


# ─────────────────────────────────────────────────────────────────────────────
# Colour helpers for 3D plot
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
    """Run one episode and return a unified flight log dict for all plots."""
    obs, _ = env.reset()

    log = dict(
        t=[], alt_m=[], alt_km=[], tilt_deg=[], tilt_x=[], tilt_y=[], psi=[],
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

        # Dead-reckon horizontal drift
        north += v_up * tilt_x * dt
        east  += v_up * tilt_y * dt

        log["t"].append(t)
        log["alt_m"].append(h_km * 1000)
        log["alt_km"].append(h_km)
        log["tilt_deg"].append(info["tilt_deg"])
        log["tilt_x"].append(tilt_x)
        log["tilt_y"].append(tilt_y)
        log["psi"].append(math.degrees(float(obs[2])))  # Yaw / unused roll depending on env setup
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

    outcome = "✓ REACHED TARGET ALTITUDE" if truncated else "✗ TERMINATED EARLY (Crashed/Diverged)"
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
# 1. 3-D Trajectory Plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_3d(log, save_path=None, arrow_every=50, phase=4, wind=0.0, turb=False, target_alt_km=25.0):
    t       = np.array(log["t"])
    alt     = np.array(log["alt_m"])
    north   = np.array(log["north"])
    east    = np.array(log["east"])
    tilt    = np.array(log["tilt_deg"])
    tilt_x  = np.array(log["tilt_x"])
    tilt_y  = np.array(log["tilt_y"])

    max_tilt = max(tilt.max(), 1.0)

    fig = plt.figure(figsize=(14, 9))
    fig.patch.set_facecolor("#0e0e12")

    gs = gridspec.GridSpec(
        2, 2, figure=fig, left=0.04, right=0.98, top=0.91, bottom=0.06, wspace=0.28, hspace=0.38,
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

    # 3D path coloured by tilt
    segment_colours = [_tilt_colour(0.5 * (tilt[i] + tilt[i + 1]), max_tilt) for i in range(len(tilt) - 1)]
    segs = _make_segments(east, north, alt)
    lc = Line3DCollection(segs, colors=segment_colours, linewidths=1.6, alpha=0.9)
    ax3d.add_collection3d(lc)

    # Nose-direction arrows
    arrow_len = max(alt.max() * 0.06, 200.0)
    for i in range(0, len(t), arrow_every):
        tx, ty = tilt_x[i], tilt_y[i]
        tz_up  = math.sqrt(max(0.0, 1.0 - tx**2 - ty**2))
        ax3d.quiver(
            east[i], north[i], alt[i],
            ty * arrow_len, tx * arrow_len, tz_up * arrow_len,
            color="#5588ff", linewidth=0.8, arrow_length_ratio=0.25, alpha=0.7,
        )

    ax3d.scatter([east[0]],  [north[0]],  [alt[0]],  s=60,  color="#22dd88", zorder=5, label="Launch")
    ax3d.scatter([east[-1]], [north[-1]], [alt[-1]], s=80,  color="#ff4455", zorder=5, label="End")
    ax3d.plot(east, north, zs=0, zdir="z", color="#334433", linewidth=0.7, alpha=0.5, linestyle="--")

    ax3d.set_xlabel("East (m)", labelpad=4)
    ax3d.set_ylabel("North (m)", labelpad=4)
    ax3d.set_zlabel("Alt (m)", labelpad=4)
    ax3d.legend(fontsize=7, facecolor="#1a1a24", edgecolor="#333344", labelcolor="#ccccdd", loc="upper left")

    # Colour bar (tilt legend)
    sm = plt.cm.ScalarMappable(
        cmap=matplotlib.colors.LinearSegmentedColormap.from_list("tilt", [(0, "#00cc66"), (0.5, "#ffaa00"), (1, "#ff2233")]),
        norm=matplotlib.colors.Normalize(vmin=0, vmax=max_tilt),
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax3d, shrink=0.55, pad=0.08, aspect=18)
    cbar.set_label("Tilt angle (°)", color="#aaaacc", fontsize=8)
    cbar.ax.yaxis.set_tick_params(color="#aaaacc", labelsize=7)
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="#aaaacc")

    # Altitude vs time
    ax_alt.plot(t, alt / 1000, color="#4499ff", linewidth=1.4)
    ax_alt.axhline(target_alt_km, color="#22dd88", linewidth=0.8, linestyle="--", alpha=0.7, label=f"Target {target_alt_km} km")
    ax_alt.fill_between(t, alt / 1000, alpha=0.12, color="#4499ff")
    ax_alt.set_xlabel("Time (s)", fontsize=8)
    ax_alt.set_ylabel("Altitude (km)", fontsize=8)
    ax_alt.set_title("Altitude", color="#ccccdd", fontsize=9, pad=4)
    ax_alt.legend(fontsize=7, facecolor="#14141a", edgecolor="#333344", labelcolor="#ccccdd")

    # Tilt vs time with danger zone (15 degrees based on Phase 2 modifications)
    ax_tlt.plot(t, tilt, color="#ffaa33", linewidth=1.4, label="Tilt°")
    ax_tlt.axhline(15.0, color="#ff4455", linewidth=0.8, linestyle="--", alpha=0.8, label="Term. limit 15°")
    ax_tlt.fill_between(t, tilt, alpha=0.15, color="#ffaa33")
    ax_tlt.set_xlabel("Time (s)", fontsize=8)
    ax_tlt.set_ylabel("Tilt (°)", fontsize=8)
    ax_tlt.set_title("Tilt angle", color="#ccccdd", fontsize=9, pad=4)
    ax_tlt.legend(fontsize=7, facecolor="#14141a", edgecolor="#333344", labelcolor="#ccccdd")

    # Title
    outcome_str = f"✓ reached {target_alt_km} km" if alt[-1] >= (target_alt_km * 1000 - 200) else f"✗ ended at {alt[-1]:.0f} m"
    fig.suptitle(
        f"3D Trajectory  ·  {outcome_str}  ·  {t[-1]:.1f} s  ·  max tilt {tilt.max():.1f}°\n"
        f"Phase {phase}  ·  Wind: {wind:.1f} m/s  ·  Turbulence: {'On' if turb else 'Off'}",
        color="#ddddee", fontsize=10, y=0.96,
    )

    if save_path:
        plt.savefig(save_path, dpi=150, facecolor=fig.get_facecolor())
        print(f"  Saved 3D plot → {save_path}")
    else:
        plt.show()

    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# 2. 2-D Telemetry Plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_flight(log, save_path=None, target_alt_km=25.0, phase=2, wind=0.0, turb=False):
    fig, axes = plt.subplots(5, 1, figsize=(10, 14), sharex=True)
    
    fig.suptitle(
        f"2D Telemetry Dashboard  ·  Phase {phase}\n"
        f"Wind: {wind:.1f} m/s  ·  Turbulence: {'On' if turb else 'Off'}",
        fontsize=14, y=0.98
    )

    # 1. Attitude: tilt from vertical + yaw heading
    axes[0].plot(log["t"], log["tilt_deg"], label="tilt (deg from vertical)", color="tab:red")
    axes[0].plot(log["t"], log["psi"], label="psi (yaw)", color="tab:purple")
    axes[0].axhline(0, color="gray", linestyle="--", linewidth=0.8)
    axes[0].axhline(15, color="tab:red", linestyle=":", linewidth=1.0, alpha=0.5, label="15° Limit")
    axes[0].set_ylabel("Angle [deg]")
    axes[0].set_title("Attitude (tilt target: 0)")
    axes[0].legend(loc="upper left")
    axes[0].grid(alpha=0.3)

    # 2. Angular rates
    axes[1].plot(log["t"], log["p"], label="p (roll rate)")
    axes[1].plot(log["t"], log["q"], label="q (pitch rate)")
    axes[1].plot(log["t"], log["r"], label="r (yaw rate)")
    axes[1].set_ylabel("Rate [deg/s]")
    axes[1].set_title("Angular rates")
    axes[1].legend(loc="upper left")
    axes[1].grid(alpha=0.3)

    # 3. Altitude / climb rate
    ax2 = axes[2].twinx()
    axes[2].plot(log["t"], log["alt_km"], color="tab:blue", label="altitude [km]")
    ax2.plot(log["t"], log["v_up"], color="tab:orange", label="v_up [m/s]", alpha=0.6)
    axes[2].axhline(
        target_alt_km, color="green", linestyle="--", linewidth=0.8,
        label=f"target {target_alt_km:.0f} km",
    )
    axes[2].set_ylabel("Altitude [km]", color="tab:blue")
    ax2.set_ylabel("v_up [m/s]", color="tab:orange")
    axes[2].set_title("Altitude and climb rate")
    axes[2].legend(loc="upper left")
    axes[2].grid(alpha=0.3)

    # 4. Thrust
    axes[3].plot(log["t"], log["thrust_n"], color="tab:red")
    axes[3].set_ylabel("Thrust [N]")
    axes[3].set_title("Engine thrust")
    axes[3].grid(alpha=0.3)

    # 5. Canard commands
    axes[4].plot(log["t"], log["a_elev"], label="elevator cmd")
    axes[4].plot(log["t"], log["a_rud"], label="rudder cmd")
    axes[4].plot(log["t"], log["a_ail"], label="aileron cmd")
    axes[4].set_ylabel("Command [-1,1]")
    axes[4].set_xlabel("Time [s]")
    axes[4].set_title("Canard commands (agent actions)")
    axes[4].legend(loc="upper left")
    axes[4].grid(alpha=0.3)

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)  # Make room for the suptitle

    if save_path:
        plt.savefig(save_path, dpi=120)
        print(f"  Saved 2D telemetry plot → {save_path}")
    else:
        plt.show()

    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run a trained PPO rocket model and generate unified 3D and 2D flight visualizations."
    )
    parser.add_argument("--model",       type=str, required=True,
                        help="Path to .zip model file, e.g. checkpoints/best_model.zip")
    
    parser.add_argument("--save-3d",     type=str, default=None,
                        help="Save the 3D plot to this file instead of displaying it")
    parser.add_argument("--save-2d",     type=str, default=None,
                        help="Save the 2D telemetry plot to this file instead of displaying it")
    
    parser.add_argument("--phase",       type=int, choices=[1, 2, 3, 4], default=4,
                        help="Select the test phase (1, 2, 3, or 4). Configures wind and turb. Default is 4.")
    
    parser.add_argument("--target-alt-km", type=float, default=25.0,
                        help="Target altitude line drawn on the plots (km). Default 25.0")
    
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

    # Define the physics parameters for each phase curriculum
    phase_configs = {
        1: {"wind": 5.0,  "turb": False},
        2: {"wind": 10.0,  "turb": True},
        3: {"wind": 15.0, "turb": True},
    }
    
    cfg = phase_configs[args.phase]

    print(f"\n  Loading model : {args.model}")
    model = PPO.load(args.model)

    env = RocketAeroJSBSimEnv(
        wind_speed_mps  = cfg["wind"],
        wind_turbulence = cfg["turb"],
    )
    if args.seed is not None:
        env.reset(seed=args.seed)

    print(f"  Test Phase    : {args.phase}")
    print(f"  Wind          : {cfg['wind']} m/s{'  (no turbulence)' if not cfg['turb'] else ' + Dryden turbulence'}")
    print(f"  Policy        : {'stochastic' if args.stochastic else 'deterministic'}")
    print(f"  Live output   : {'off' if args.no_live else f'every {args.live_every} steps'}")
    print()

    # Run the single simulation to capture telemetry
    log, truncated = run_episode(
        model, env,
        deterministic = not args.stochastic,
        live          = not args.no_live,
        live_every    = args.live_every,
    )

    env.close()

    print("  Rendering plots...")
    
    # Render Plot 1: 3D Trajectory
    plot_3d(
        log, 
        save_path=args.save_3d, 
        arrow_every=args.arrow_every,
        phase=args.phase,
        wind=cfg["wind"],
        turb=cfg["turb"],
        target_alt_km=args.target_alt_km
    )

    # Render Plot 2: 2D Telemetry
    plot_flight(
        log, 
        save_path=args.save_2d, 
        target_alt_km=args.target_alt_km,
        phase=args.phase,
        wind=cfg["wind"],
        turb=cfg["turb"]
    )


if __name__ == "__main__":
    main()
