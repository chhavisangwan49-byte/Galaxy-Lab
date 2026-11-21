"""
GALAXY LAB - Step 5: Central Mass ("Black Hole") Experiment
--------------------------------------------------------------
Question: what happens to the galaxy when the central mass grows?

We keep the disk of stars exactly the same and make the central mass
heavier and heavier (10 -> 25 -> 50 -> 100). Then we compare the
rotation curves and check how stable each galaxy stays.
"""

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------
# 1. SETTINGS
# ---------------------------
N_STARS = 200
R_MIN = 0.5
GALAXY_RADIUS = 5.0
STAR_MASS = 0.1                  # total disk mass = N_STARS * STAR_MASS = 20

G = 1.0
eps = 0.1
dt = 0.005
n_steps = 3000

CENTRAL_MASSES = [10.0, 25.0, 50.0, 100.0]     # the thing we vary
DISK_MASS = N_STARS * STAR_MASS                # = 20 for every run

# ---------------------------
# 2. PHYSICS (vectorized leapfrog, same as before)
# ---------------------------
def compute_acceleration(pos, mass, G, eps):
    diff = pos[None, :, :] - pos[:, None, :]
    dist_sq = np.sum(diff**2, axis=2) + eps**2
    inv_d3 = dist_sq ** -1.5
    np.fill_diagonal(inv_d3, 0.0)
    return G * np.sum(mass[None, :, None] * diff * inv_d3[:, :, None], axis=1)

def leapfrog_step(pos, vel, mass, dt, G, eps):
    acc = compute_acceleration(pos, mass, G, eps)
    vel_half = vel + acc * dt / 2
    pos_new = pos + vel_half * dt
    acc_new = compute_acceleration(pos_new, mass, G, eps)
    vel_new = vel_half + acc_new * dt / 2
    return pos_new, vel_new

def total_energy(pos, vel, mass, G, eps):
    ke = np.sum(0.5 * mass * np.sum(vel**2, axis=1))
    diff = pos[None, :, :] - pos[:, None, :]
    r = np.sqrt(np.sum(diff**2, axis=2) + eps**2)
    inv_r = 1.0 / r
    np.fill_diagonal(inv_r, 0.0)
    pe = -0.5 * G * np.sum(mass[:, None] * mass[None, :] * inv_r)
    return ke + pe

# ---------------------------
# 3. ONE EXPERIMENT
# ---------------------------
def run_experiment(M_c):

    np.random.seed(42)   # same star layout every run -> only the central mass differs
    radii = R_MIN + (GALAXY_RADIUS - R_MIN) * np.sqrt(np.random.uniform(0, 1, N_STARS))
    angles = np.random.uniform(0, 2 * np.pi, N_STARS)

    order = np.argsort(radii)
    enclosed = np.empty(N_STARS)
    enclosed[order] = M_c + STAR_MASS * np.arange(N_STARS)

    speeds = np.sqrt(G * enclosed * radii**2 / (radii**2 + eps**2) ** 1.5)

    pos = np.zeros((N_STARS + 1, 2))
    vel = np.zeros((N_STARS + 1, 2))
    mass = np.full(N_STARS + 1, STAR_MASS)
    mass[0] = M_c
    pos[1:, 0] = radii * np.cos(angles)
    pos[1:, 1] = radii * np.sin(angles)
    vel[1:, 0] = -speeds * np.sin(angles)
    vel[1:, 1] = speeds * np.cos(angles)

    E0 = total_energy(pos, vel, mass, G, eps)
    snap_r, snap_v = [], []
    for step in range(n_steps):
        pos, vel = leapfrog_step(pos, vel, mass, dt, G, eps)
        if step > 0.6 * n_steps and step % 100 == 0:
            rel_p = pos[1:] - pos[0]
            rel_v = vel[1:] - vel[0]
            snap_r.append(np.sqrt(np.sum(rel_p**2, axis=1)))
            snap_v.append(np.sqrt(np.sum(rel_v**2, axis=1)))
    E1 = total_energy(pos, vel, mass, G, eps)

    drift = abs((E1 - E0) / E0) * 100
    snap_r = np.concatenate(snap_r)
    snap_v = np.concatenate(snap_v)
    inside = np.mean(snap_r < 2 * GALAXY_RADIUS) * 100

    # Prediction from the starting setup: v = sqrt(G * M_enclosed / r)
    r_grid = np.linspace(R_MIN, 7.0, 200)
    enc_grid = np.interp(r_grid, radii[order], enclosed[order])
    v_pred = np.sqrt(G * enc_grid * r_grid**2 / (r_grid**2 + eps**2) ** 1.5)
    return snap_r, snap_v, r_grid, v_pred, drift, inside

# ---------------------------
# 4. RUN ALL
# ---------------------------
bins = np.linspace(R_MIN, 7.0, 11)
bin_mid = 0.5 * (bins[:-1] + bins[1:])
colors = ["tab:green", "tab:blue", "tab:orange", "tab:red"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
inner_speeds = []

print(f"{'Central mass':>12} | {'% of galaxy':>11} | {'Energy drift':>12} | {'Stars within 2R':>15} | {'Prediction error':>16}")
print("-" * 80)

for run_no, (M_c, color) in enumerate(zip(CENTRAL_MASSES, colors), start=1):
    pct = M_c / (M_c + DISK_MASS) * 100
    print(f"...running experiment {run_no}/{len(CENTRAL_MASSES)} (central mass = {M_c:.0f}), please wait...")
    snap_r, snap_v, r_grid, v_pred, drift, inside = run_experiment(M_c)

    med = np.full(len(bin_mid), np.nan)
    for i in range(len(bin_mid)):
        sel = (snap_r >= bins[i]) & (snap_r < bins[i + 1])
        if sel.sum() >= 5:
            med[i] = np.median(snap_v[sel])

    pred_at_bins = np.interp(bin_mid, r_grid, v_pred)
    ok = ~np.isnan(med)
    pred_err = np.mean(np.abs(med[ok] - pred_at_bins[ok]) / pred_at_bins[ok]) * 100

    inner = np.median(snap_v[(snap_r >= R_MIN) & (snap_r < 2.0)])
    inner_speeds.append(inner)

    ax1.plot(r_grid, v_pred, '--', color=color, alpha=0.6)
    ax1.plot(bin_mid[ok], med[ok], 'o-', color=color, label=f"Central mass = {M_c:.0f} ({pct:.0f}% of galaxy)")
    print(f"{M_c:>12.0f} | {pct:>10.0f}% | {drift:>10.3f} % | {inside:>13.1f} % | {pred_err:>14.1f} %")

ax1.set_xlabel("Radius from center (r)")
ax1.set_ylabel("Orbital speed (v)")
ax1.set_title("Rotation curves for different central masses\n(solid = simulation, dashed = prediction)")
ax1.set_xlim(0, 7)
ax1.legend()
ax1.grid(alpha=0.3)

ax2.bar([f"{m:.0f}" for m in CENTRAL_MASSES], inner_speeds, color=colors)
ax2.set_xlabel("Central mass")
ax2.set_ylabel("Typical orbital speed of inner stars (r < 2)")
ax2.set_title("Heavier center -> faster inner orbits")
ax2.grid(alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig("central_mass_experiment.png", dpi=150)
print("Saved plot as central_mass_experiment.png")
plt.show() 