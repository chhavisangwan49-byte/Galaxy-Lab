import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ---------------------------
# 1. UNITS (normalized, not SI)
# ---------------------------
# We use "sim units" instead of real kg/meters/seconds, because real
# astrophysical numbers are either huge (masses) or tiny (many quantities
# in SI), which causes numerical headaches. In sim units we just SET:
G = 1.0          # gravitational constant = 1
eps = 0.05       # softening length (prevents divide-by-zero at close range)
dt = 0.01        # time step size
n_steps = 4000   # how many steps to simulate

# ---------------------------
# 2. INITIAL CONDITIONS: two stars
# ---------------------------
# positions: shape (2 stars, 2 coordinates [x, y])
pos = np.array([
    [ 1.0, 0.0],   # star A starts here
    [-1.0, 0.0],   # star B starts here
])

# velocities: chosen so they orbit each other (not fall straight in)
vel = np.array([
    [0.0,  0.5],
    [0.0, -0.5],
])

mass = np.array([1.0, 1.0])  # equal-mass stars

# ---------------------------
# 3. THE PHYSICS: compute gravity on each star
# ---------------------------
def compute_acceleration(pos, mass, G, eps):
    """
    For every star, add up the gravitational pull from every OTHER star.
    Vectorized with numpy so there's no slow Python for-loop over stars
    (this is what lets this scale to 1000+ stars later).
    """
    n = pos.shape[0]
    acc = np.zeros_like(pos)

    for i in range(n):
        # vector from star i to every other star j
        diff = pos - pos[i]                      # shape (n, 2)
        dist_sq = np.sum(diff**2, axis=1) + eps**2  # softened distance^2
        dist_cube = dist_sq**1.5

        # avoid a star pulling on itself (i == i term)
        dist_cube[i] = np.inf

        # F = G * m_j * (vector direction) / dist^3, summed over all j
        acc[i] = G * np.sum((mass[:, None] * diff) / dist_cube[:, None], axis=0)

    return acc

# ---------------------------
# 4. THE LEAPFROG STEP
# ---------------------------
def leapfrog_step(pos, vel, mass, dt, G, eps):
    acc = compute_acceleration(pos, mass, G, eps)
    vel_half = vel + acc * dt / 2          # half-step velocity kick
    pos_new = pos + vel_half * dt          # full position drift
    acc_new = compute_acceleration(pos_new, mass, G, eps)
    vel_new = vel_half + acc_new * dt / 2  # second half-step kick
    return pos_new, vel_new

# ---------------------------
# 5. RUN THE SIMULATION & STORE HISTORY (for animation + energy check)
# ---------------------------
history = np.zeros((n_steps, 2, 2))  # (steps, stars, xy)
energy_history = np.zeros(n_steps)

def total_energy(pos, vel, mass, G, eps):
    # Kinetic energy: 0.5 * m * v^2, summed over stars
    ke = np.sum(0.5 * mass * np.sum(vel**2, axis=1))
    # Potential energy: -G*m1*m2 / distance, summed over unique pairs
    pe = 0.0
    n = pos.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            r = np.sqrt(np.sum((pos[i] - pos[j])**2) + eps**2)
            pe += -G * mass[i] * mass[j] / r
    return ke + pe

for step in range(n_steps):
    history[step] = pos
    energy_history[step] = total_energy(pos, vel, mass, G, eps)
    pos, vel = leapfrog_step(pos, vel, mass, dt, G, eps)

# ---------------------------
# 6. ANIMATE IT
# ---------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

ax1.set_xlim(-2.5, 2.5)
ax1.set_ylim(-2.5, 2.5)
ax1.set_aspect('equal')
ax1.set_title("Two-Body Orbit")
trail_a, = ax1.plot([], [], '-', color='orange', alpha=0.4, lw=1)
trail_b, = ax1.plot([], [], '-', color='cyan', alpha=0.4, lw=1)
star_a, = ax1.plot([], [], 'o', color='orange', markersize=10)
star_b, = ax1.plot([], [], 'o', color='cyan', markersize=10)

ax2.set_title("Total Energy vs Time (should stay ~flat)")
ax2.set_xlim(0, n_steps)
energy_line, = ax2.plot([], [], color='lime')
ax2.set_ylim(energy_history.min() * 1.2, energy_history.max() * 0.8)

def animate(frame):
    trail_len = 300
    start = max(0, frame - trail_len)
    trail_a.set_data(history[start:frame, 0, 0], history[start:frame, 0, 1])
    trail_b.set_data(history[start:frame, 1, 0], history[start:frame, 1, 1])
    star_a.set_data([history[frame, 0, 0]], [history[frame, 0, 1]])
    star_b.set_data([history[frame, 1, 0]], [history[frame, 1, 1]])
    energy_line.set_data(np.arange(frame), energy_history[:frame])
    return trail_a, trail_b, star_a, star_b, energy_line

ani = animation.FuncAnimation(
    fig, animate, frames=n_steps, interval=10, blit=True
)

plt.tight_layout()
plt.show()