"""
visualize_flight.py
====================
Run one episode with a trained PPO model and plot the resulting
flight trajectory: attitude (tilt + yaw), rates, altitude, thrust,
and canard commands over time.

Usage:
    python visualize_flight.py --model ./checkpoints/best_model.zip
    python visualize_flight.py --model ./checkpoints/best_model.zip --save flight.png
    python visualize_flight.py --model ./checkpoints/best_model.zip --wind 0 --no-turb
"""

import argparse
import math
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

from rocket_env_jsbsim import RocketAeroJSBSimEnv


def run_episode(model, env, deterministic=True):
    obs, _ = env.reset()
    log = {
        "t": [], "tilt_deg": [], "tilt_x": [], "tilt_y": [], "psi": [],
        "p": [], "q": [], "r": [],
        "h": [], "v_up": [], "thrust": [],
        "a_elev": [], "a_rud": [], "a_ail": [],
    }

    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        log["t"].append(info["sim_time"])
        log["tilt_deg"].append(info["tilt_deg"])
        log["tilt_x"].append(float(obs[0]))
        log["tilt_y"].append(float(obs[1]))
        log["psi"].append(math.degrees(obs[2]))
        log["p"].append(math.degrees(obs[3]))
        log["q"].append(math.degrees(obs[4]))
        log["r"].append(math.degrees(obs[5]))
        log["h"].append(obs[6])
        log["v_up"].append(obs[7])
        log["thrust"].append(info["thrust_n"])
        log["a_elev"].append(action[0])
        log["a_rud"].append(action[1])
        log["a_ail"].append(action[2])

    outcome = "REACHED TARGET" if truncated else "TERMINATED EARLY (crashed/diverged)"
    print(f"Episode finished after {log['t'][-1]:.2f}s — {outcome}")
    print(f"Max altitude: {max(log['h']):.3f} km")
    print(
        f"Max tilt={max(log['tilt_deg']):.2f}deg  "
        f"max |psi|={max(abs(p) for p in log['psi']):.2f}deg"
    )

    return log


def plot_flight(log, save_path=None, target_alt_km=40.0):
    fig, axes = plt.subplots(5, 1, figsize=(10, 14), sharex=True)

    # Attitude: tilt from vertical + yaw heading
    axes[0].plot(log["t"], log["tilt_deg"], label="tilt (deg from vertical)", color="tab:red")
    axes[0].plot(log["t"], log["psi"], label="psi (yaw)", color="tab:purple")
    axes[0].axhline(0, color="gray", linestyle="--", linewidth=0.8)
    axes[0].set_ylabel("Angle [deg]")
    axes[0].set_title("Attitude (tilt target: 0)")
    axes[0].legend(loc="upper right")
    axes[0].grid(alpha=0.3)

    # Angular rates
    axes[1].plot(log["t"], log["p"], label="p (roll rate)")
    axes[1].plot(log["t"], log["q"], label="q (pitch rate)")
    axes[1].plot(log["t"], log["r"], label="r (yaw rate)")
    axes[1].set_ylabel("Rate [deg/s]")
    axes[1].set_title("Angular rates")
    axes[1].legend(loc="upper right")
    axes[1].grid(alpha=0.3)

    # Altitude / climb rate
    ax2 = axes[2].twinx()
    axes[2].plot(log["t"], log["h"], color="tab:blue", label="altitude [km]")
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

    # Thrust
    axes[3].plot(log["t"], log["thrust"], color="tab:red")
    axes[3].set_ylabel("Thrust [N]")
    axes[3].set_title("Engine thrust")
    axes[3].grid(alpha=0.3)

    # Canard commands
    axes[4].plot(log["t"], log["a_elev"], label="elevator cmd")
    axes[4].plot(log["t"], log["a_rud"], label="rudder cmd")
    axes[4].plot(log["t"], log["a_ail"], label="aileron cmd")
    axes[4].set_ylabel("Command [-1,1]")
    axes[4].set_xlabel("Time [s]")
    axes[4].set_title("Canard commands (agent actions)")
    axes[4].legend(loc="upper right")
    axes[4].grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=120)
        print(f"Saved plot to {save_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="path to .zip model")
    parser.add_argument("--save", type=str, default=None, help="save plot instead of showing it")
    parser.add_argument("--wind", type=float, default=10.0, help="Max steady crosswind [m/s] (default 10)")
    parser.add_argument("--no-turb", action="store_true", help="Disable Dryden turbulence")
    parser.add_argument("--stochastic", action="store_true", help="sample actions instead of greedy")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for env reset")
    parser.add_argument(
        "--target-alt-km", type=float, default=40.0,
        help="Target altitude line drawn on the plot (km)",
    )
    args = parser.parse_args()

    env = RocketAeroJSBSimEnv(
        wind_speed_mps=args.wind,
        wind_turbulence=not args.no_turb,
    )
    if args.seed is not None:
        env.reset(seed=args.seed)

    model = PPO.load(args.model)

    log = run_episode(model, env, deterministic=not args.stochastic)
    env.close()

    plot_flight(log, save_path=args.save, target_alt_km=args.target_alt_km)


if __name__ == "__main__":
    main()
