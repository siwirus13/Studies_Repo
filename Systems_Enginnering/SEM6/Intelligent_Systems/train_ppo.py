"""
train_ppo.py
===============
PPO training for RocketAeroJSBSimEnv with:
  - 4-phase wind curriculum (scaled for 40 km ceiling)
  - Entropy decay schedule across phases
  - RocketProgressCallback: per-episode altitude / tilt / outcome printout

Usage:
    python train_ppo_v2.py --timesteps 3000000
    python train_ppo_v2.py --timesteps 3000000 --resume checkpoints/ppo_rocket_final.zip

Monitor in TensorBoard:
    tensorboard --logdir ./logs
"""

import argparse
import os
import sys
import numpy as np
from dataclasses import dataclass
from typing import Optional

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
    CallbackList,
)
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import get_linear_fn

from rocket_env_jsbsim import RocketAeroJSBSimEnv


# ─────────────────────────────────────────────────────────────────────────────
# Curriculum phases
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Phase:
    name:        str
    timesteps:   int            # steps to train in this phase
    wind_mps:    float          # max steady crosswind [m/s]
    turbulence:  bool
    ic_perturb:  float          # std of initial angle/rate perturbation (rad)
    ent_coef:    float          # entropy coefficient for this phase
    description: str = ""


PHASES = [
    Phase(
        name        = "Phase 1 — steady crosswind",
        timesteps   = 4_000_000,
        wind_mps    = 5.0,
        turbulence  = True,
        ic_perturb  = 0.05,
        ent_coef    = 0.02, # Start tighter since wind forces immediate adjustments
        description = "Steady 5 m/s crosswind. Learn to counter continuous drift.",
    ),
    Phase(
        name        = "Phase 2 — wind + turbulence",
        timesteps   = 3_000_000,
        wind_mps    = 10.0,
        turbulence  = True,
        ic_perturb  = 0.08,
        ent_coef    = 0.01,
        description = "10 m/s wind + Dryden turbulence. Damping structural oscillations.",
    ),
    Phase(
        name        = "Phase 3 — full chaos",
        timesteps   = 2_500_000,
        wind_mps    = 15.0,
        turbulence  = True,
        ic_perturb  = 0.12,
        ent_coef    = 0.002,
        description = "Max structural survival validation.",
    ),
]

# ─────────────────────────────────────────────────────────────────────────────
# Progress callback
# ─────────────────────────────────────────────────────────────────────────────

class RocketProgressCallback(BaseCallback):
    """
    Prints a summary line for every completed episode across all envs.

    Columns:
      step      — global timestep
      ep        — episode index (this phase)
      outcome   — REACH (hit 40 km) | CRASH (tilt/spin/ground) | TIMEOUT
      max_alt   — highest altitude reached [m]
      ep_len    — episode length [steps]
      mean_tilt — mean tilt angle over the episode [deg]
      reward    — total episode reward
    """

    OUTCOME_REACH   = "✓ REACH  "
    OUTCOME_CRASH   = "✗ CRASH  "
    OUTCOME_TIMEOUT = "~ TIMEOUT"

    def __init__(self, print_every: int = 1, verbose: int = 1):
        super().__init__(verbose)
        self.print_every = print_every   # print every N completed episodes

        # Per-env accumulators  (filled once n_envs is known)
        self._ep_reward:   Optional[np.ndarray] = None
        self._ep_steps:    Optional[np.ndarray] = None
        self._ep_max_alt:  Optional[np.ndarray] = None
        self._ep_tilt_sum: Optional[np.ndarray] = None

        self._ep_count   = 0
        self._phase_name = ""

        # Rolling stats for the print_every window
        self._window_rewards:  list[float] = []
        self._window_max_alts: list[float] = []
        self._window_tilts:    list[float] = []
        self._window_lengths:  list[int]   = []
        self._window_outcomes: list[str]   = []

    def set_phase(self, phase_name: str):
        """Call before each phase's model.learn()."""
        self._phase_name = phase_name
        self._ep_count   = 0
        self._window_rewards.clear()
        self._window_max_alts.clear()
        self._window_tilts.clear()
        self._window_lengths.clear()
        self._window_outcomes.clear()

    def _init_accumulators(self):
        n = self.training_env.num_envs
        self._ep_reward   = np.zeros(n, dtype=np.float64)
        self._ep_steps    = np.zeros(n, dtype=np.int64)
        self._ep_max_alt  = np.zeros(n, dtype=np.float64)
        self._ep_tilt_sum = np.zeros(n, dtype=np.float64)

    def _on_training_start(self):
        self._init_accumulators()
        print()
        print("═" * 90)
        print(f"  {self._phase_name}")
        print("═" * 90)
        print(f"  {'Step':>10}  {'Ep':>5}  {'Outcome':12}  "
              f"{'MaxAlt':>9}  {'EpLen':>6}  {'MeanTilt':>9}  {'TotalRew':>10}")
        print("─" * 90)

    def _on_step(self) -> bool:
        rewards  = self.locals["rewards"]          # shape (n_envs,)
        dones    = self.locals["dones"]            # shape (n_envs,)
        infos    = self.locals["infos"]            # list of dicts

        for i, (r, done, info) in enumerate(zip(rewards, dones, infos)):
            self._ep_reward[i]   += r
            self._ep_steps[i]    += 1

            alt_m = info.get("altitude_m", 0.0)
            tilt_deg = info.get("tilt_deg", 0.0)

            self._ep_max_alt[i]  = max(self._ep_max_alt[i], alt_m)
            self._ep_tilt_sum[i] += tilt_deg

            if done:
                self._ep_count += 1
                ep_len = int(self._ep_steps[i])

                # Determine outcome from environment thresholds
                truncated = info.get("TimeLimit.truncated", False)
                if truncated:
                    outcome = self.OUTCOME_TIMEOUT
                # FIX: Scaled up threshold to 39,800m to match the environment's new 40 km ceiling targets
                elif alt_m >= 39_800.0:          
                    outcome = self.OUTCOME_REACH
                else:
                    outcome = self.OUTCOME_CRASH

                mean_tilt = self._ep_tilt_sum[i] / max(ep_len, 1)
                total_rew = self._ep_reward[i]
                max_alt   = self._ep_max_alt[i]

                self._window_outcomes.append(outcome)
                self._window_rewards.append(total_rew)
                self._window_max_alts.append(max_alt)
                self._window_tilts.append(mean_tilt)
                self._window_lengths.append(ep_len)

                # Print every N episodes
                if self._ep_count % self.print_every == 0:
                    n = len(self._window_outcomes)
                    reach_pct = self._window_outcomes.count(self.OUTCOME_REACH) / n * 100
                    crash_pct = self._window_outcomes.count(self.OUTCOME_CRASH) / n * 100

                    if self.print_every > 1:
                        avg_alt  = np.mean(self._window_max_alts)
                        avg_len  = np.mean(self._window_lengths)
                        avg_tilt = np.mean(self._window_tilts)
                        avg_rew  = np.mean(self._window_rewards)
                        print(
                            f"  {self.num_timesteps:>10,}  "
                            f"{self._ep_count:>5}  "
                            f"R{reach_pct:3.0f}% C{crash_pct:3.0f}%  "
                            f"{avg_alt:>8.0f}m  "
                            f"{avg_len:>6.0f}  "
                            f"{avg_tilt:>8.1f}°  "
                            f"{avg_rew:>10.1f}"
                        )
                    else:
                        print(
                            f"  {self.num_timesteps:>10,}  "
                            f"{self._ep_count:>5}  "
                            f"{outcome}  "
                            f"{max_alt:>8.0f}m  "
                            f"{ep_len:>6}  "
                            f"{mean_tilt:>8.1f}°  "
                            f"{total_rew:>10.1f}"
                        )
                    # Clear window
                    self._window_rewards.clear()
                    self._window_max_alts.clear()
                    self._window_tilts.clear()
                    self._window_lengths.clear()
                    self._window_outcomes.clear()

                # Reset this env's accumulators
                self._ep_reward[i]   = 0.0
                self._ep_steps[i]    = 0
                self._ep_max_alt[i]  = 0.0
                self._ep_tilt_sum[i] = 0.0

        return True

    def _on_training_end(self):
        print("─" * 90)
        print(f"  Phase complete at step {self.num_timesteps:,}  |  "
              f"Total episodes this phase: {self._ep_count}")
        print()


# ─────────────────────────────────────────────────────────────────────────────
# Env factory
# ─────────────────────────────────────────────────────────────────────────────

def make_env(phase: Phase):
    def _init():
        env = RocketAeroJSBSimEnv(
            wind_speed_mps  = phase.wind_mps,
            wind_turbulence = phase.turbulence,
            max_altitude_m  = 40_000.0  # Explicitly force 40 km ceiling limits
        )
        env._ic_perturb = phase.ic_perturb
        env = Monitor(env)
        return env
    return _init


# ─────────────────────────────────────────────────────────────────────────────
# Main Execution Block
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_envs",   type=int, default=4)
    parser.add_argument("--logdir",   type=str, default="./logs")
    parser.add_argument("--savedir",  type=str, default="./checkpoints")
    parser.add_argument("--resume",   type=str, default=None,
                        help="path to .zip to resume from (skips Phase 1 warmup)")
    parser.add_argument("--start_phase", type=int, default=1,
                        help="which phase to start from (1-indexed, default 1)")
    args = parser.parse_args()

    # FIX: Safety Guard Clause preventing advanced curriculum starts with an untrained agent
    if args.start_phase > 1 and not args.resume:
        print(f"ERROR: Cannot initialize training at Phase {args.start_phase} without establishing base reflexes.", file=sys.stderr)
        print("Please supply an explicit baseline model configuration using the '--resume' argument.", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.logdir,  exist_ok=True)
    os.makedirs(args.savedir, exist_ok=True)

    progress_cb = RocketProgressCallback(print_every=10, verbose=0)
    model = None

    for phase_idx, phase in enumerate(PHASES, start=1):
        if phase_idx < args.start_phase:
            print(f"Skipping {phase.name}")
            continue

        print(f"\n{'━'*90}")
        print(f"  Starting {phase.name}")
        print(f"  {phase.description}")
        print(f"  Steps: {phase.timesteps:,}  |  "
              f"Wind: {phase.wind_mps} m/s  |  "
              f"Turbulence: {phase.turbulence}  |  "
              f"ent_coef: {phase.ent_coef}")
        print(f"{'━'*90}")

        # Note: Re-creating SubprocVecEnv per phase is safer here than dynamically changing attributes.
        # It guarantees the underlying JSBSim C++ memory states are entirely flushed out between heavy environment shifts.
        env = SubprocVecEnv([make_env(phase) for _ in range(args.n_envs)])
        eval_env = Monitor(RocketAeroJSBSimEnv(
            wind_speed_mps  = phase.wind_mps,
            wind_turbulence = phase.turbulence,
            max_altitude_m  = 30_000.0
        ))

        # Learning rate: linear decay within each phase block
        lr_schedule = get_linear_fn(3e-4, 5e-5, 1.0)

        if model is None:
            if args.resume:
                print(f"  Loading model baseline weights from {args.resume}")
                # phase2_best was saved under the env's old observation space
                # (alt ceiling 12 km, before the 40 km curriculum). PPO.load()
                # enforces exact Box-bound equality when an env is passed, which
                # trips here even though obs/action *shapes* are unchanged.
                # Load without an env (skips the space check), then transplant
                # weights into a freshly built model that uses the new env.
                old_model = PPO.load(args.resume, device="auto")

                model = PPO(
                    "MlpPolicy",
                    env,
                    verbose=0,
                    tensorboard_log=args.logdir,
                    n_steps=2048,
                    batch_size=256,
                    learning_rate=lr_schedule,
                    gamma=0.995,
                    gae_lambda=0.95,
                    clip_range=0.2,
                    ent_coef=phase.ent_coef,
                    vf_coef=0.5,
                    max_grad_norm=0.5,
                    n_epochs=10,
                    policy_kwargs=dict(net_arch=[256, 256]),
                )
                model.set_parameters(old_model.get_parameters())
            else:
                model = PPO(
                    "MlpPolicy",
                    env,
                    verbose=0,          
                    tensorboard_log=args.logdir,
                    n_steps=2048,
                    batch_size=256,
                    learning_rate=lr_schedule,
                    gamma=0.995,
                    gae_lambda=0.95,
                    clip_range=0.2,
                    ent_coef=phase.ent_coef,
                    vf_coef=0.5,
                    max_grad_norm=0.5,
                    n_epochs=10,
                    policy_kwargs=dict(net_arch=[256, 256]),
                )
        else:
            # Reassign the newly spawned vector environments to the model parameters
            model.set_env(env)
            model.ent_coef = phase.ent_coef

        checkpoint_cb = CheckpointCallback(
            save_freq=max(50_000 // args.n_envs, 1),
            save_path=args.savedir,
            name_prefix=f"ppo_rocket_p{phase_idx}",
        )
        eval_cb = EvalCallback(
            eval_env,
            best_model_save_path=os.path.join(args.savedir, f"phase{phase_idx}_best"),
            log_path=args.logdir,
            eval_freq=max(25_000 // args.n_envs, 1),
            n_eval_episodes=5,
            deterministic=True,
            verbose=0,
        )

        progress_cb.set_phase(phase.name)

        # FIX: Unified tb_log_name across all loop instances preserves continuous trendline graphing
        model.learn(
            total_timesteps=phase.timesteps,
            callback=CallbackList([progress_cb, checkpoint_cb, eval_cb]),
            tb_log_name="ppo_rocket_curriculum",
            reset_num_timesteps=False,
        )

        phase_path = os.path.join(args.savedir, f"ppo_rocket_phase{phase_idx}_final")
        model.save(phase_path)
        print(f"  Saved phase checkpoint: {phase_path}.zip")

        env.close()
        eval_env.close()

    final_path = os.path.join(args.savedir, "ppo_rocket_final")
    model.save(final_path)
    print(f"\n{'━'*90}")
    print(f"  All phases complete. Final unified model: {final_path}.zip")
    print(f"{'━'*90}\n")


if __name__ == "__main__":
    main()
