"""
GALAXY LAB - Step 4: Dark Matter Experiment
----------------------------------------------
We add an invisible dark matter halo around the galaxy. We don't
simulate dark matter particles -- we just add its gravitational pull
as a formula (a "logarithmic halo").

Then we run the SAME galaxy several times with different halo
strengths and compare their rotation curves:

    no halo      -> curve falls with radius (like the solar system)
    bigger halo  -> curve flattens (like REAL galaxies)

Before each run the program also PREDICTS the curve from a formula,
then compares prediction vs. simulation ("Predict the Galaxy").
"""

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------
# 1. SETTINGS
# ---------------------------
N_STARS = 200
R_MIN = 0.5
GALAXY_RADIUS = 5.0
CENTRAL_MASS = 100.0
STAR_MASS = 0.1

G = 1.0
eps = 0.1
dt = 0.005
n_steps = 3000

# Dark matter halo settings
HALO_CORE = 2.0                      # core radius of the halo
HALO_STRENGTHS = [0.0, 4.0, 6.0, 8.0]  # V_halo: halo's "flat" rotation speed. 0 = no dark matter

# ---------------------------
# 2. HELPER FUNCTIONS
# ---------------------------
def halo_speed_sq(r, V_halo):
    """Contribution of the halo to v^2 at radius r (flattens to V_halo^2 far away)."""
    return V_halo**2 * r**2 / (r**2 + HALO_CORE**2)

def compute_acceleration(pos, mass, G, eps, V_halo):
    # Star-star + star-center gravity (vectorized, no Python loop)
    diff = pos[None, :, :] - pos[:, None, :]
    dist_sq = np.sum(diff**2, axis=2) + eps**2
    inv_d3 = dist_sq ** -1.5
    np.fill_diagonal(inv_d3, 0.0)
    acc = G * np.sum(mass[None, :, None] * diff * inv_d3[:, :, None], axis=1)
    # Dark matter halo: pulls every body toward the origin
    r2 = np.sum(pos**2, axis=1, keepdims=True)
    acc += -V_halo**2 * pos / (r2 + HALO_CORE**2)
    return acc

def leapfrog_step(pos, vel, mass, dt, G, eps, V_halo):
    acc = compute_acceleration(pos, mass, G, eps, V_halo)
    vel_half = vel + acc * dt / 2
    pos_new = pos + vel_half * dt
    acc_new = compute_acceleration(pos_new, mass, G, eps, V_halo)
    vel_new = vel_half + acc_new * dt / 2
    return pos_new, vel_new

def total_energy(pos, vel, mass, G, eps, V_halo):
    ke = np.sum(0.5 * mass * np.sum(vel**2, axis=1))
    diff = pos[None, :, :] - pos[:, None, :]
    r = np.sqrt(np.sum(diff**2, axis=2) + eps**2)
    inv_r = 1.0 / r
    np.fill_diagonal(inv_r, 0.0)
    pe = -0.5 * G * np.sum(mass[:, None] * mass[None, :] * inv_r)
    # halo potential energy: 0.5 * V^2 * ln(r^2 + core^2) per unit mass
    r2 = np.sum(pos**2, axis=1)
    pe += np.sum(mass * 0.5 * V_halo**2 * np.log(r2 + HALO_CORE**2))
    return ke + pe

# ---------------------------
# 3. ONE EXPERIMENT = ONE FULL SIMULATION
# ---------------------------
def run_experiment(V_halo):
    np.random.seed(42)   # same galaxy every time, so only the halo differs
    radii = R_MIN + (GALAXY_RADIUS - R_MIN) * np.sqrt(np.random.uniform(0, 1, N_STARS))
    angles = np.random.uniform(0, 2 * np.pi, N_STARS)

    order = np.argsort(radii)
    enclosed = np.empty(N_STARS)
    enclosed[order] = CENTRAL_MASS + STAR_MASS * np.arange(N_STARS)

    # Starting speed = visible-matter gravity + halo gravity
    v2_visible = G * enclosed * radii**2 / (radii**2 + eps**2) ** 1.5
    speeds = np.sqrt(v2_visible + halo_speed_sq(radii, V_halo))

    pos = np.zeros((N_STARS + 1, 2))
    vel = np.zeros((N_STARS + 1, 2))
    mass = np.full(N_STARS + 1, STAR_MASS)
    mass[0] = CENTRAL_MASS
    pos[1:, 0] = radii * np.cos(angles)
    pos[1:, 1] = radii * np.sin(angles)
    vel[1:, 0] = -speeds * np.sin(angles)
    vel[1:, 1] = speeds * np.cos(angles)

    E0 = total_energy(pos, vel, mass, G, eps, V_halo)
    snap_r, snap_v = [], []
    for step in range(n_steps):
        pos, vel = leapfrog_step(pos, vel, mass, dt, G, eps, V_halo)
        # collect snapshots in the last 40% of the run (averages out the orbit-to-orbit scatter)
        if step > 0.6 * n_steps and step % 100 == 0:
            rel_p = pos[1:] - pos[0]
            rel_v = vel[1:] - vel[0]
            snap_r.append(np.sqrt(np.sum(rel_p**2, axis=1)))
            snap_v.append(np.sqrt(np.sum(rel_v**2, axis=1)))
    E1 = total_energy(pos, vel, mass, G, eps, V_halo)

    drift = abs((E1 - E0) / E0) * 100
    snap_r = np.concatenate(snap_r)
    snap_v = np.concatenate(snap_v)
    inside = np.mean(snap_r < 2 * GALAXY_RADIUS) * 100

    # Prediction: visible mass (interpolated from the starting setup) + halo
    r_grid = np.linspace(R_MIN, 7.0, 200)
    enc_grid = np.interp(r_grid, radii[order], enclosed[order])
    v_pred = np.sqrt(G * enc_grid * r_grid**2 / (r_grid**2 + eps**2) ** 1.5
                     + halo_speed_sq(r_grid, V_halo))
    return snap_r, snap_v, r_grid, v_pred, drift, inside

# ---------------------------
# 4. RUN ALL EXPERIMENTS
# ---------------------------
bins = np.linspace(R_MIN, 7.0, 11)
bin_mid = 0.5 * (bins[:-1] + bins[1:])
colors = ["gray", "tab:blue", "tab:green", "tab:red"]

plt.figure(figsize=(9, 6))
print(f"{'Halo V':>8} | {'Energy drift':>12} | {'Stars within 2R':>15} | {'Prediction error':>16}")
print("-" * 62)

for run_no, (V_halo, color) in enumerate(zip(HALO_STRENGTHS, colors), start=1):
    print(f"...running experiment {run_no}/{len(HALO_STRENGTHS)} (halo V = {V_halo}), please wait...")
    snap_r, snap_v, r_grid, v_pred, drift, inside = run_experiment(V_halo)

    # Median speed in each radius bin (robust to a few wild stars)
    med = np.full(len(bin_mid), np.nan)
    for i in range(len(bin_mid)):
        sel = (snap_r >= bins[i]) & (snap_r < bins[i + 1])
        if sel.sum() >= 5:
            med[i] = np.median(snap_v[sel])

    # Prediction vs actual, at the bin centers
    pred_at_bins = np.interp(bin_mid, r_grid, v_pred)
    ok = ~np.isnan(med)
    pred_err = np.mean(np.abs(med[ok] - pred_at_bins[ok]) / pred_at_bins[ok]) * 100

    label = "No dark matter" if V_halo == 0 else f"Halo V = {V_halo}"
    plt.plot(r_grid, v_pred, '--', color=color, alpha=0.6)
    plt.plot(bin_mid[ok], med[ok], 'o-', color=color, label=label)
    print(f"{V_halo:>8.1f} | {drift:>10.3f} % | {inside:>13.1f} % | {pred_err:>14.1f} %")

plt.xlabel("Radius from center (r)")
plt.ylabel("Orbital speed (v)")
plt.title("Effect of a Dark Matter Halo on the Rotation Curve\n(solid = simulation, dashed = prediction)")
plt.xlim(0, 7)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("dark_matter_curves.png", dpi=150)
print("Saved plot as dark_matter_curves.png")
plt.show()