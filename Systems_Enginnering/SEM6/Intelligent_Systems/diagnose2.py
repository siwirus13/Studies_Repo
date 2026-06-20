"""
diagnose2.py
============
Same as diagnose.py but with wind/turbulence enabled and RANDOM actions
(matching what PPO does during early exploration), to reproduce the
step-3/4 termination seen during training.
"""
import math
import numpy as np
from rocket_env_jsbsim import RocketAeroJSBSimEnv

env = RocketAeroJSBSimEnv(
    render_mode=None,
    debug=True,
    wind_turbulence=True,
    wind_speed_mps=10.0,
)

def safe_get(fdm, prop):
    try:
        return fdm.get_property_value(prop)
    except Exception:
        return float("nan")

for seed in range(5):
    print(f"\n\n========== SEED {seed} ==========")
    obs, _ = env.reset(seed=seed)
    fdm = env._fdm

    wn = safe_get(fdm, "atmosphere/wind-north-fps")
    we = safe_get(fdm, "atmosphere/wind-east-fps")
    print(f"  Initial wind: N={wn:.2f} fps  E={we:.2f} fps")

    rng = np.random.default_rng(seed)
    for i in range(15):
        # Sample actions the way PPO would early on: roughly N(0,1) clipped
        action = np.clip(rng.normal(0, 1.0, size=3), -1, 1).astype(np.float32)
        obs, reward, terminated, truncated, info = env.step(action)

        phi, theta_dev, psi, p, q, r, h_km, v_up = obs

        thrust_mag = safe_get(fdm, "external_reactions/thrust/magnitude")
        qbar = safe_get(fdm, "aero/qbar-psf")
        alpha = safe_get(fdm, "aero/alpha-rad")
        beta = safe_get(fdm, "aero/beta-rad")
        elev = safe_get(fdm, "fcs/elevator-pos-rad")
        rud  = safe_get(fdm, "fcs/rudder-pos-rad")
        ail  = safe_get(fdm, "fcs/left-aileron-pos-rad")

        print(f"  step {i+1:2d} t={fdm.get_sim_time():.3f}s "
              f"act=[{action[0]:+.2f},{action[1]:+.2f},{action[2]:+.2f}] "
              f"phi={math.degrees(phi):+7.2f} theta_dev={math.degrees(theta_dev):+7.2f} "
              f"psi={math.degrees(psi):+7.2f} | "
              f"p={math.degrees(p):+6.1f} q={math.degrees(q):+6.1f} r={math.degrees(r):+6.1f} | "
              f"h={h_km*1000:7.2f}m v_up={v_up:6.1f} | "
              f"alpha={math.degrees(alpha):+7.2f} beta={math.degrees(beta):+7.2f} qbar={qbar:7.2f} | "
              f"canard(e,r,a)=({math.degrees(elev):+5.1f},{math.degrees(rud):+5.1f},{math.degrees(ail):+5.1f}) | "
              f"r={reward:+8.2f} term={terminated}")

        if terminated or truncated:
            print(f"  *** ENDED at step {i+1} ({'TRUNC' if truncated else 'TERM'}) ***")
            break

env.close()
