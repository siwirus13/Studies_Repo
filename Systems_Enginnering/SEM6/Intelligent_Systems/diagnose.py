"""
diagnose.py
===========
Step through the rocket env one JSBSim timestep at a time, printing
every relevant property, to find why the episode terminates almost
immediately.
"""
import math
from rocket_env_jsbsim import RocketAeroJSBSimEnv
import numpy as np

env = RocketAeroJSBSimEnv(render_mode=None, debug=True, wind_turbulence=False, wind_speed_mps=0.0)
obs, _ = env.reset(seed=0)

fdm = env._fdm

print("\n=== Property catalog check: external_reactions / thrust / force ===")
catalog = fdm.get_property_catalog()
if isinstance(catalog, str):
    catalog = catalog.split("\n")

matched = False
for line in catalog:
    if any(kw in line.lower() for kw in ["external_reactions", "thrust", "fbx", "fby", "fbz"]):
        print(" ", line.strip())
        matched = True
if not matched:
    print("  (no matches found - printing first 40 entries of full catalog instead)")
    for line in catalog[:40]:
        if line.strip():
            print("  ", line.strip())

def safe_get(fdm, prop):
    try:
        return fdm.get_property_value(prop)
    except Exception:
        return float("nan")


print("\n=== Stepping manually with zero action, full diagnostics ===")
for i in range(10):
    action = np.zeros(3, dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)

    phi, theta_dev, psi, p, q, r, h_km, v_up = obs

    fbx = safe_get(fdm, "forces/fbx-total-lbs")
    fby = safe_get(fdm, "forces/fby-total-lbs")
    fbz = safe_get(fdm, "forces/fbz-total-lbs")
    mx  = safe_get(fdm, "moments/l-total-lbsft")
    my  = safe_get(fdm, "moments/m-total-lbsft")
    mz  = safe_get(fdm, "moments/n-total-lbsft")
    thrust_mag = safe_get(fdm, "external_reactions/thrust/magnitude")
    qbar = safe_get(fdm, "aero/qbar-psf")
    alpha = safe_get(fdm, "aero/alpha-rad")
    beta = safe_get(fdm, "aero/beta-rad")

    print(f"\n--- step {i+1}  t={fdm.get_sim_time():.3f}s ---")
    print(f"  phi={math.degrees(phi):+8.3f} theta_dev={math.degrees(theta_dev):+8.3f} psi={math.degrees(psi):+8.3f} deg")
    print(f"  p={math.degrees(p):+8.3f} q={math.degrees(q):+8.3f} r={math.degrees(r):+8.3f} deg/s")
    print(f"  h={h_km*1000:.2f} m   v_up={v_up:.3f} m/s")
    print(f"  thrust_mag={thrust_mag:.2f} lbf")
    print(f"  Forces (body, lbf): Fx={fbx:.3f} Fy={fby:.3f} Fz={fbz:.3f}")
    print(f"  Moments (body, lbf-ft): L={mx:.4f} M={my:.4f} N={mz:.4f}")
    print(f"  qbar={qbar:.4f} psf   alpha={math.degrees(alpha):.3f} deg  beta={math.degrees(beta):.3f} deg")
    print(f"  reward={reward:.3f}  terminated={terminated}  truncated={truncated}")

    if terminated or truncated:
        print("\n*** EPISODE ENDED ***")
        break

env.close()
