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
n_steps = 4000

# ---------------------------
# 2. GENERATE THE GALAXY
# ---------------------------
np.random.seed(42)

radii = R_MIN + (GALAXY_RADIUS - R_MIN) * np.sqrt(np.random.uniform(0, 1, N_STARS))
angles = np.random.uniform(0, 2 * np.pi, N_STARS)

order = np.argsort(radii)
enclosed = np.empty(N_STARS)
enclosed[order] = CENTRAL_MASS + STAR_MASS * np.arange(N_STARS)

speeds = np.sqrt(G * enclosed * radii**2 / (radii**2 + eps**2) ** 1.5)

pos = np.zeros((N_STARS + 1, 2))
vel = np.zeros((N_STARS + 1, 2))
mass = np.full(N_STARS + 1, STAR_MASS)
mass[0] = CENTRAL_MASS

pos[1:, 0] = radii * np.cos(angles)
pos[1:, 1] = radii * np.sin(angles)
vel[1:, 0] = -speeds * np.sin(angles)
vel[1:, 1] = speeds * np.cos(angles)

initial_radii = radii.copy()
initial_speeds = speeds.copy()

# ---------------------------
# 3. PHYSICS
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
# 4. RUN
# ---------------------------
E_start = total_energy(pos, vel, mass, G, eps)
for step in range(n_steps):
    pos, vel = leapfrog_step(pos, vel, mass, dt, G, eps)
    if step % 500 == 0:
        print(f"Step {step}/{n_steps}")
E_end = total_energy(pos, vel, mass, G, eps)

print("Simulation done, measuring rotation curve...")

# ---------------------------
# 5. MEASURE
# ---------------------------
rel_pos = pos[1:] - pos[0]
rel_vel = vel[1:] - vel[0]
final_radii = np.sqrt(np.sum(rel_pos**2, axis=1))
final_speeds = np.sqrt(np.sum(rel_vel**2, axis=1))

drift = abs((E_end - E_start) / E_start) * 100
inside = np.mean(final_radii < 2 * GALAXY_RADIUS) * 100
print(f"Energy drift:                 {drift:.3f} %")
print(f"Stars still inside 2x radius: {inside:.1f} %")

# ---------------------------
# 6. PLOT
# ---------------------------
r_theory = np.linspace(R_MIN, GALAXY_RADIUS, 200)
v_theory = np.sqrt(G * CENTRAL_MASS * r_theory**2 / (r_theory**2 + eps**2) ** 1.5)

plt.figure(figsize=(8, 6))
plt.scatter(initial_radii, initial_speeds, s=10, color="gray", alpha=0.4, label="Stars at start")
plt.scatter(final_radii, final_speeds, s=12, color="dodgerblue", alpha=0.7, label="Stars at end")
plt.plot(r_theory, v_theory, '--', color="orange", label="Prediction (central mass only)")
plt.xlim(0, GALAXY_RADIUS * 1.5)
plt.xlabel("Radius from center (r)")
plt.ylabel("Orbital speed (v)")
plt.title(f"Galaxy Rotation Curve (N = {N_STARS} stars, {n_steps} steps)\n"
          f"Energy drift {drift:.2f}%, {inside:.0f}% of stars still bound")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("rotation_curve.png", dpi=150)
print("Saved plot as rotation_curve.png")
plt.show()