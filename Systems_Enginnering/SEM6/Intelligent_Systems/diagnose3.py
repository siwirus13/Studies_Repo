"""
diagnose3.py
============
Same idea as diagnose2.py (wind + random actions, mimicking PPO's early
exploration) but updated for the new gimbal-lock-free observation space:
[tilt_x, tilt_y, roll_unused, p, q, r, h_km, v_up].
"""
import math
import numpy as np
from rocket_env_jsbsim import RocketAeroJSBSimEnv

env = RocketAeroJSBSimEnv(
    render_mode=None,
    debug=False,
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

    rng = np.random.default_rng(seed)
    for i in range(300):
        action = np.clip(rng.normal(0, 1.0, size=3), -1, 1).astype(np.float32)
        obs, reward, terminated, truncated, info = env.step(action)

        tilt_x, tilt_y, _roll, p, q, r, h_km, v_up = obs
        tilt_deg = math.degrees(math.asin(min(1.0, math.sqrt(tilt_x**2 + tilt_y**2))))

        qbar = safe_get(fdm, "aero/qbar-psf")
        elev = safe_get(fdm, "fcs/elevator-pos-rad")
        rud  = safe_get(fdm, "fcs/rudder-pos-rad")
        ail  = safe_get(fdm, "fcs/left-aileron-pos-rad")

        print(f"  step {i+1:2d} t={fdm.get_sim_time():.3f}s "
              f"act=[{action[0]:+.2f},{action[1]:+.2f},{action[2]:+.2f}] "
              f"tilt={tilt_deg:6.2f}deg (x={tilt_x:+.3f} y={tilt_y:+.3f}) | "
              f"p={math.degrees(p):+6.1f} q={math.degrees(q):+6.1f} r={math.degrees(r):+6.1f} deg/s | "
              f"h={h_km*1000:7.2f}m v_up={v_up:6.1f} | "
              f"qbar={qbar:7.2f} | "
              f"canard(e,r,a)=({math.degrees(elev):+5.1f},{math.degrees(rud):+5.1f},{math.degrees(ail):+5.1f}) | "
              f"r={reward:+8.2f} term={terminated}")

        if terminated or truncated:
            print(f"  *** ENDED at step {i+1} ({'TRUNC' if truncated else 'TERM'}) ***")
            break

env.close()
