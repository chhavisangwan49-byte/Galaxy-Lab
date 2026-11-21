"""
GALAXY LAB - Step 6: Computational Scaling Experiment
--------------------------------------------------------
Question: how does the run time grow when we add more stars?

Direct N-body gravity makes every star feel every other star, so the
work per time step should grow like N^2 (double the stars -> 4x the work).
We measure the real run time for N = 100 ... 2000 stars and check this.

We also PREDICT the run time for the biggest galaxy from the small ones,
then compare prediction vs. actual.
"""

import time
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------
# 1. SETTINGS
# ---------------------------
N_VALUES = [100, 250, 500, 1000, 2000]
R_MIN = 0.5
GALAXY_RADIUS = 5.0
CENTRAL_MASS = 100.0
STAR_MASS = 0.1

G = 1.0
eps = 0.1
dt = 0.005

STEPS_TO_TIME = 10      # we time just a few steps and average (no need to run the whole simulation)
REPEATS = 3             # repeat and keep the fastest (removes noise from other programs on your laptop)
FULL_RUN_STEPS = 3000   # used to project "how long would a full run take?"

# ---------------------------
# 2. PHYSICS (same as before)
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

def make_galaxy(n_stars):
    np.random.seed(42)
    radii = R_MIN + (GALAXY_RADIUS - R_MIN) * np.sqrt(np.random.uniform(0, 1, n_stars))
    angles = np.random.uniform(0, 2 * np.pi, n_stars)
    order = np.argsort(radii)
    enclosed = np.empty(n_stars)
    enclosed[order] = CENTRAL_MASS + STAR_MASS * np.arange(n_stars)
    speeds = np.sqrt(G * enclosed * radii**2 / (radii**2 + eps**2) ** 1.5)

    pos = np.zeros((n_stars + 1, 2))
    vel = np.zeros((n_stars + 1, 2))
    mass = np.full(n_stars + 1, STAR_MASS)
    mass[0] = CENTRAL_MASS
    pos[1:, 0] = radii * np.cos(angles)
    pos[1:, 1] = radii * np.sin(angles)
    vel[1:, 0] = -speeds * np.sin(angles)
    vel[1:, 1] = speeds * np.cos(angles)
    return pos, vel, mass

# ---------------------------
# 3. TIME EACH GALAXY SIZE
# ---------------------------
def time_per_step(n_stars):
    pos, vel, mass = make_galaxy(n_stars)
    pos, vel = leapfrog_step(pos, vel, mass, dt, G, eps)     # warm-up (first call is always slower)
    best = np.inf
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        for _ in range(STEPS_TO_TIME):
            pos, vel = leapfrog_step(pos, vel, mass, dt, G, eps)
        elapsed = (time.perf_counter() - t0) / STEPS_TO_TIME
        best = min(best, elapsed)
    return best

print(f"{'Stars (N)':>10} | {'Time per step':>14} | {'Full run (3000 steps)':>22}")
print("-" * 55)

times = []
for n in N_VALUES:
    t = time_per_step(n)
    times.append(t)
    full_run = t * FULL_RUN_STEPS
    shown = f"{full_run:.1f} s" if full_run < 120 else f"{full_run/60:.1f} min"
    print(f"{n:>10} | {t*1000:>11.2f} ms | {shown:>22}")

N = np.array(N_VALUES, dtype=float)
T = np.array(times)

# ---------------------------
# 4. HOW DOES TIME GROW? (fit  time = c * N^p  on a log-log scale)
# ---------------------------
p, log_c = np.polyfit(np.log(N), np.log(T), 1)
print(f"\nFitted scaling:  time  ~  N^{p:.2f}   (pure direct summation predicts N^2)")

# Predict the biggest run from the smaller ones assuming exactly N^2
small = N[:-1]
c_est = np.mean(T[:-1] / small**2)
pred_big = c_est * N[-1]**2
err = abs(pred_big - T[-1]) / T[-1] * 100
print(f"Predicted time for N = {int(N[-1])} (from smaller runs, assuming N^2): {pred_big*1000:.1f} ms")
print(f"Actual time:                                                    {T[-1]*1000:.1f} ms")
print(f"Prediction error:                                               {err:.1f} %")

# ---------------------------
# 5. PLOT
# ---------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

ref = T[0] * (N / N[0])**2
ax1.loglog(N, T * 1000, 'o-', color="tab:blue", label=f"Measured  (fit: N^{p:.2f})")
ax1.loglog(N, ref * 1000, '--', color="orange", label="Pure N^2 reference")
ax1.set_xlabel("Number of stars (N)")
ax1.set_ylabel("Time per step (ms)")
ax1.set_title("Run time vs number of stars (log-log)")
ax1.set_xticks(N_VALUES)
ax1.set_xticklabels([str(n) for n in N_VALUES])
ax1.legend()
ax1.grid(alpha=0.3, which="both")

full_minutes = T * FULL_RUN_STEPS / 60
ax2.bar([str(n) for n in N_VALUES], full_minutes, color="tab:purple")
ax2.set_xlabel("Number of stars (N)")
ax2.set_ylabel(f"Time for a full {FULL_RUN_STEPS}-step run (minutes)")
ax2.set_title("What it means in practice")
ax2.grid(alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig("scaling_experiment.png", dpi=150)
print("Saved plot as scaling_experiment.png")
plt.show()