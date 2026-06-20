"""
diagnose_env.py
===============
Run this locally (no training, just physics inspection) to find out
why episodes are stuck at 42 steps / 235m regardless of the policy.

Usage:
    python diagnose_env.py

It will tell you exactly which of the five root causes is the problem.
"""

import math
import numpy as np
from rocket_env_jsbsim import RocketAeroJSBSimEnv, FT_TO_M, LBF_TO_N

# ─────────────────────────────────────────────────────────────────────────────
# TEST 1: Open-loop with ZERO action and ZERO IC perturbation
#         A perfectly initialised rocket with no control should stay upright
#         if the plant is stable, or reveal how fast it destabilises if not.
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("TEST 1: Zero IC perturbation, zero action, zero wind")
print("        Expected: rocket climbs straight up indefinitely")
print("        If it crashes: plant is aerodynamically unstable beyond recovery")
print("=" * 70)

env = RocketAeroJSBSimEnv(wind_speed_mps=0.0, wind_turbulence=False, debug=False)
env._ic_perturb = 0.0   # perfect initial condition
obs, _ = env.reset(seed=0)

# Manually zero out ALL perturbations after reset
env._fdm.set_property_value("attitude/phi-rad",   0.0)
env._fdm.set_property_value("attitude/theta-rad", math.pi / 2.0)
env._fdm.set_property_value("attitude/psi-rad",   0.0)
env._fdm.set_property_value("velocities/p-rad_sec", 0.0)
env._fdm.set_property_value("velocities/q-rad_sec", 0.0)
env._fdm.set_property_value("velocities/r-rad_sec", 0.0)
env._fdm.run_ic()
obs = env._get_obs()

print(f"\nInitial state:")
print(f"  tilt_x={obs[0]:+.6f}  tilt_y={obs[1]:+.6f}  (should be ~0)")
print(f"  p={obs[3]:+.6f}  q={obs[4]:+.6f}  r={obs[5]:+.6f}  (should be 0)")
print(f"  alt={obs[6]*1000:.1f}m  v_up={obs[7]:.2f} m/s")
print()

zero_action = np.zeros(3, dtype=np.float32)
print(f"{'Step':>5}  {'Alt(m)':>8}  {'v_up':>7}  {'tilt°':>7}  "
      f"{'q(rad/s)':>10}  {'Fx_lbf':>10}  {'Fz_lbf':>10}  {'Cm_lbft':>10}")
print("-" * 80)

crashed_at = None
for i in range(500):
    obs, r, terminated, truncated, info = env.step(zero_action)

    tilt_deg = info['tilt_deg']
    alt_m    = info['altitude_m']
    v_up     = obs[7]
    q_rate   = obs[4]

    # Raw forces and moments — the smoking gun
    fx  = env._fdm.get_property_value("forces/fbx-total-lbs")
    fz  = env._fdm.get_property_value("forces/fbz-total-lbs")
    cm  = env._fdm.get_property_value("moments/m-total-lbsft")
    alpha = math.degrees(env._fdm.get_property_value("aero/alpha-rad"))

    if i < 10 or i % 25 == 0 or terminated or truncated:
        print(f"{i:>5}  {alt_m:>8.1f}  {v_up:>7.1f}  {tilt_deg:>7.2f}  "
              f"{q_rate:>10.4f}  {fx:>10.2f}  {fz:>10.2f}  {cm:>10.2f}")

    if terminated or truncated:
        crashed_at = i
        reason = 'TRUNCATED (reached altitude!)' if truncated else 'TERMINATED'
        print(f"\n>>> Episode ended at step {i}: {reason}")
        break

env.close()

if crashed_at and crashed_at < 100:
    print("\n⚠ DIAGNOSIS: Plant crashes even with PERFECT initial conditions.")
    print("  The aerodynamic model is numerically or physically unstable.")
    print("  Check: aircraft XML mass, CG/CP location, Cm_alpha sign.")
elif crashed_at is None:
    print("\n✓ Rocket survived 500 steps with zero perturbation. Plant is stable.")

# ─────────────────────────────────────────────────────────────────────────────
# TEST 2: Thrust direction check
#         Thrust via external_reactions should push the rocket along its body
#         X-axis (nose direction). When pointing straight up, Fz in local NED
#         should be strongly negative (upward). Check it.
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("TEST 2: Thrust direction — is it actually pushing upward?")
print("=" * 70)

env = RocketAeroJSBSimEnv(wind_speed_mps=0.0, wind_turbulence=False, debug=False)
env._ic_perturb = 0.0
obs, _ = env.reset(seed=0)
env._fdm.set_property_value("attitude/theta-rad", math.pi / 2.0)
env._fdm.set_property_value("velocities/p-rad_sec", 0.0)
env._fdm.set_property_value("velocities/q-rad_sec", 0.0)
env._fdm.set_property_value("velocities/r-rad_sec", 0.0)
env._fdm.run_ic()

# One step with thrust explicitly set
thrust_lbf = env.thrust_lbf
env._fdm.set_property_value("external_reactions/thrust/magnitude", thrust_lbf)
env._fdm.run()

print(f"\n  Thrust set: {thrust_lbf:.1f} lbf = {thrust_lbf * LBF_TO_N:.0f} N")
print(f"  forces/fbx-total-lbs = {env._fdm.get_property_value('forces/fbx-total-lbs'):+.2f}")
print(f"  forces/fby-total-lbs = {env._fdm.get_property_value('forces/fby-total-lbs'):+.2f}")
print(f"  forces/fbz-total-lbs = {env._fdm.get_property_value('forces/fbz-total-lbs'):+.2f}")
print(f"  (In JSBSim body frame: X=nose, Z=down. Thrust should appear as large +Fx)")
print(f"  velocities/v-down-fps = {env._fdm.get_property_value('velocities/v-down-fps'):+.4f}")
print(f"  (Negative v-down = moving upward ✓)")

env.close()

# ─────────────────────────────────────────────────────────────────────────────
# TEST 3: Canard authority — can full deflection overcome a 10° tilt?
#         Set the rocket at 10° tilt with zero rates, apply full elevator.
#         Watch whether pitch rate (q) goes negative (corrective) or positive.
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("TEST 3: Canard authority — does full elevator produce corrective moment?")
print("        Rocket at 10° tilt, full elevator deflection")
print("        Expected: q goes negative (nose corrects back toward vertical)")
print("=" * 70)

env = RocketAeroJSBSimEnv(wind_speed_mps=0.0, wind_turbulence=False, debug=False)
env._ic_perturb = 0.0
obs, _ = env.reset(seed=0)

# Set 10° tilt in pitch (theta = 90° - 10° = 80°)
tilt_angle = math.radians(80)   # theta
env._fdm.set_property_value("attitude/theta-rad", tilt_angle)
env._fdm.set_property_value("velocities/p-rad_sec", 0.0)
env._fdm.set_property_value("velocities/q-rad_sec", 0.0)
env._fdm.set_property_value("velocities/r-rad_sec", 0.0)
env._fdm.run_ic()

print(f"\n  Initial tilt: 10°, full elevator (+1.0)")
print(f"  {'Step':>5}  {'tilt°':>8}  {'q(rad/s)':>10}  {'Cm(lbsft)':>12}  {'note'}")
print(f"  {'-'*60}")

corrective = []
full_elev = np.array([1.0, 0.0, 0.0], dtype=np.float32)

for i in range(20):
    obs, r, terminated, truncated, info = env.step(full_elev)
    tilt_deg = info['tilt_deg']
    q_rate   = obs[4]
    cm       = env._fdm.get_property_value("moments/m-total-lbsft")
    corrective.append(q_rate)
    note = "← correcting" if q_rate < 0 else "← WRONG WAY"
    print(f"  {i:>5}  {tilt_deg:>8.2f}  {q_rate:>10.4f}  {cm:>12.2f}  {note}")
    if terminated or truncated:
        print(f"  Episode ended at step {i}")
        break

env.close()

net_q = sum(corrective)
print(f"\n  Sum of q over 20 steps: {net_q:+.4f}")
if net_q < 0:
    print("  ✓ Canards ARE producing corrective moment")
else:
    print("  ⚠ Canards are NOT corrective — wrong sign or insufficient authority")
    print("    Check: elevator Cm derivative sign in aircraft XML")

# ─────────────────────────────────────────────────────────────────────────────
# TEST 4: What does the mass / T-W ratio look like?
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("TEST 4: Mass and thrust-to-weight ratio")
print("=" * 70)

env = RocketAeroJSBSimEnv(wind_speed_mps=0.0, wind_turbulence=False, debug=False)
env._ic_perturb = 0.0
obs, _ = env.reset(seed=0)

mass_slug = env._fdm.get_property_value("inertia/mass-slugs")
mass_kg   = mass_slug * 14.5939
weight_n  = mass_kg * 9.81
tw_ratio  = env.thrust_lbf * LBF_TO_N / weight_n

print(f"\n  mass:        {mass_slug:.4f} slugs = {mass_kg:.1f} kg")
print(f"  weight:      {weight_n:.1f} N")
print(f"  thrust:      {env.thrust_lbf * LBF_TO_N:.0f} N")
print(f"  T/W ratio:   {tw_ratio:.2f}")
if tw_ratio < 1.0:
    print(f"  ⚠ T/W < 1: ROCKET CANNOT LIFT OFF. Increase thrust or reduce mass.")
elif tw_ratio < 1.5:
    print(f"  ⚠ T/W is marginal. Rocket can lift but barely — any tilt kills climb.")
else:
    print(f"  ✓ T/W looks reasonable")

ixx = env._fdm.get_property_value("inertia/ixx-slugs_ft2")
iyy = env._fdm.get_property_value("inertia/iyy-slugs_ft2")
izz = env._fdm.get_property_value("inertia/izz-slugs_ft2")
print(f"\n  Moments of inertia:")
print(f"    Ixx (roll):  {ixx:.4f} slug·ft²")
print(f"    Iyy (pitch): {iyy:.4f} slug·ft²")
print(f"    Izz (yaw):   {izz:.4f} slug·ft²")

env.close()

print()
print("=" * 70)
print("SUMMARY — look for ⚠ warnings above to find the root cause")
print("=" * 70)
