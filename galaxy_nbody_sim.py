import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ---------------------------
# 1. SETTINGS - play with these!
# ---------------------------
N_STARS = 2000          # try 100, 200, 500... see how it feels on your laptop
GALAXY_RADIUS = 5.0    # how spread out the disk is
CENTRAL_MASS = 500.0   # mass of the central "black hole" (pulls everything into orbit)
STAR_MASS = 1.0        # mass of each individual star (kept equal for simplicity)

G = 1.0
eps = 0.1              # softening length (bigger than before since we have more stars)
dt = 0.01
n_steps = 5000

# ---------------------------
# 2. GENERATE A GALAXY (disk of stars around a central mass)
# ---------------------------
np.random.seed(42)  # so results are reproducible while testing

# Random radius and angle for each star -> converted to x, y
radii = GALAXY_RADIUS * np.sqrt(np.random.uniform(0, 1, N_STARS))
angles = np.random.uniform(0, 2 * np.pi, N_STARS)

star_x = radii * np.cos(angles)
star_y = radii * np.sin(angles)

# Circular orbital velocity around the central mass:
# v = sqrt(G * M_enclosed / r)  -- gives roughly stable circular orbits
speeds = np.sqrt(G * CENTRAL_MASS / (radii + eps))
# velocity direction perpendicular to the radius vector (for orbit, not fall-in)
vel_x = -speeds * np.sin(angles)
vel_y = speeds * np.cos(angles)

# Now build full arrays including the central mass as "star 0"
pos = np.zeros((N_STARS + 1, 2))
vel = np.zeros((N_STARS + 1, 2))
mass = np.zeros(N_STARS + 1)

pos[0] = [0.0, 0.0]         # central mass sits at the origin
vel[0] = [0.0, 0.0]         # starts stationary
mass[0] = CENTRAL_MASS

pos[1:, 0] = star_x
pos[1:, 1] = star_y
vel[1:, 0] = vel_x
vel[1:, 1] = vel_y
mass[1:] = STAR_MASS

n_bodies = N_STARS + 1

# ---------------------------
# 3. PHYSICS (identical structure to the 2-body version)
# ---------------------------
def compute_acceleration(pos, mass, G, eps):
    n = pos.shape[0]
    acc = np.zeros_like(pos)
    for i in range(n):
        diff = pos - pos[i]
        dist_sq = np.sum(diff**2, axis=1) + eps**2
        dist_cube = dist_sq**1.5
        dist_cube[i] = np.inf
        acc[i] = G * np.sum((mass[:, None] * diff) / dist_cube[:, None], axis=0)
    return acc

def leapfrog_step(pos, vel, mass, dt, G, eps):
    acc = compute_acceleration(pos, mass, G, eps)
    vel_half = vel + acc * dt / 2
    pos_new = pos + vel_half * dt
    acc_new = compute_acceleration(pos_new, mass, G, eps)
    vel_new = vel_half + acc_new * dt / 2
    return pos_new, vel_new

# ---------------------------
# 4. RUN SIMULATION (store only every few steps to keep memory small)
# ---------------------------
save_every = 2   # store every 2nd step -> smaller history, faster animation
n_saved = n_steps // save_every

history = np.zeros((n_saved, n_bodies, 2))

save_idx = 0
for step in range(n_steps):
    if step % save_every == 0:
        history[save_idx] = pos
        save_idx += 1
    pos, vel = leapfrog_step(pos, vel, mass, dt, G, eps)
    if step % 200 == 0:
        print(f"Step {step}/{n_steps}")   # progress indicator, since this takes longer than 2-body

print("Simulation done, rendering animation...")

# ---------------------------
# 5. ANIMATE
# ---------------------------
fig, ax = plt.subplots(figsize=(7, 7))
ax.set_xlim(-GALAXY_RADIUS * 1.5, GALAXY_RADIUS * 1.5)
ax.set_ylim(-GALAXY_RADIUS * 1.5, GALAXY_RADIUS * 1.5)
ax.set_aspect('equal')
ax.set_facecolor('black')
ax.set_title(f"Galaxy Lab - {N_STARS} stars")

stars_plot, = ax.plot([], [], 'o', color='white', markersize=1.5)
center_plot, = ax.plot([], [], 'o', color='orange', markersize=8)

def animate(frame):
    stars_plot.set_data(history[frame, 1:, 0], history[frame, 1:, 1])
    center_plot.set_data([history[frame, 0, 0]], [history[frame, 0, 1]])
    return stars_plot, center_plot

ani = animation.FuncAnimation(
    fig, animate, frames=n_saved, interval=20, blit=True
)

plt.tight_layout()
plt.show()
