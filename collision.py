"""
GALAXYLAB - STEP 7
THE COSMIC ENCOUNTER
A cinematic galaxy collision simulation.

Features:
- Two interacting galaxies
- 202 gravitational bodies
- Leapfrog integration
- Energy and momentum diagnostics
- Glowing particles
- Star trails
- Cosmic background
- Six cinematic snapshots
- Animated collision
- Scientific dashboard
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import to_rgba


# ==========================================================
# 1. SETTINGS
# ==========================================================

N_STARS = 100

R_MIN = 1.0
GALAXY_RADIUS = 5.0

CENTRAL_MASS_1 = 50.0
MASS_RATIO = 1.0
STAR_MASS = 0.05

SEPARATION = 25.0
V_REL = 2.0
IMPACT = 6.0

SPIN_1 = +1
SPIN_2 = +1

G = 1.0

eps = 0.1
EPS_CORE = 1.0

dt = 0.01
n_steps = 6000
SAVE_EVERY = 20

LIM = 30

BG = "#050713"
BLUE = "#28C9FF"
GOLD = "#FFB547"
WHITE = "#F3F5FF"
PURPLE = "#A78BFA"
MUTED = "#8994B5"


# ==========================================================
# 2. CREATE A GALAXY
# ==========================================================

def make_galaxy(central_mass, spin, seed):

    rng = np.random.default_rng(seed)

    radii = R_MIN + (
        GALAXY_RADIUS - R_MIN
    ) * np.sqrt(rng.uniform(0, 1, N_STARS))

    angles = rng.uniform(0, 2 * np.pi, N_STARS)

    order = np.argsort(radii)

    stars_inside = np.empty(N_STARS)
    stars_inside[order] = np.arange(N_STARS)

    v2 = (
        G * central_mass * radii**2
        / (radii**2 + EPS_CORE**2)**1.5
        +
        G * STAR_MASS * stars_inside * radii**2
        / (radii**2 + eps**2)**1.5
    )

    speeds = np.sqrt(v2)

    pos = np.zeros((N_STARS + 1, 2))
    vel = np.zeros((N_STARS + 1, 2))

    mass = np.full(N_STARS + 1, STAR_MASS)
    mass[0] = central_mass

    pos[1:, 0] = radii * np.cos(angles)
    pos[1:, 1] = radii * np.sin(angles)

    vel[1:, 0] = -spin * speeds * np.sin(angles)
    vel[1:, 1] = spin * speeds * np.cos(angles)

    return pos, vel, mass


# ==========================================================
# 3. BUILD TWO GALAXIES
# ==========================================================

pos1, vel1, mass1 = make_galaxy(
    CENTRAL_MASS_1, SPIN_1, 1
)

pos2, vel2, mass2 = make_galaxy(
    CENTRAL_MASS_1 * MASS_RATIO, SPIN_2, 2
)

M1_total = mass1.sum()
M2_total = mass2.sum()

v1_bulk = V_REL * M2_total / (M1_total + M2_total)
v2_bulk = V_REL * M1_total / (M1_total + M2_total)

pos1 += np.array([-SEPARATION / 2, -IMPACT / 2])
pos2 += np.array([+SEPARATION / 2, +IMPACT / 2])

vel1 += np.array([+v1_bulk, 0.0])
vel2 += np.array([-v2_bulk, 0.0])

pos = np.vstack([pos1, pos2])
vel = np.vstack([vel1, vel2])
mass = np.concatenate([mass1, mass2])

core_idx = [0, N_STARS + 1]

soft = np.full(len(mass), eps)
soft[core_idx] = EPS_CORE

EPS2 = np.maximum(
    soft[:, None],
    soft[None, :]
)**2

stars_1 = np.arange(1, N_STARS + 1)

stars_2 = np.arange(
    N_STARS + 2,
    2 * N_STARS + 2
)


# ==========================================================
# 4. PHYSICS ENGINE
# ==========================================================

def compute_acceleration(pos, mass):

    diff = pos[None, :, :] - pos[:, None, :]

    dist_sq = np.sum(diff**2, axis=2) + EPS2

    inv_d3 = dist_sq**-1.5

    np.fill_diagonal(inv_d3, 0.0)

    acc = G * np.sum(
        mass[None, :, None]
        * diff
        * inv_d3[:, :, None],
        axis=1
    )

    return acc


def leapfrog_step(pos, vel, mass, dt):

    acc = compute_acceleration(pos, mass)

    vel_half = vel + acc * dt / 2

    pos_new = pos + vel_half * dt

    acc_new = compute_acceleration(pos_new, mass)

    vel_new = vel_half + acc_new * dt / 2

    return pos_new, vel_new


def total_energy(pos, vel, mass):

    ke = np.sum(
        0.5 * mass * np.sum(vel**2, axis=1)
    )

    diff = pos[None, :, :] - pos[:, None, :]

    r = np.sqrt(
        np.sum(diff**2, axis=2) + EPS2
    )

    inv_r = 1.0 / r

    np.fill_diagonal(inv_r, 0.0)

    pe = -0.5 * G * np.sum(
        mass[:, None] * mass[None, :] * inv_r
    )

    return ke + pe


def total_momentum(vel, mass):

    return np.sum(
        mass[:, None] * vel,
        axis=0
    )


def angular_momentum(pos, vel, mass):

    return np.sum(
        mass * (
            pos[:, 0] * vel[:, 1]
            - pos[:, 1] * vel[:, 0]
        )
    )


# ==========================================================
# 5. RUN SIMULATION
# ==========================================================

print("\nGALAXYLAB INITIALIZED")
print("Preparing gravitational calculations...\n")

n_saved = n_steps // SAVE_EVERY + 1

history = np.zeros(
    (n_saved, len(mass), 2)
)

times = np.zeros(n_saved)
energy = np.zeros(n_saved)
momentum = np.zeros(n_saved)
ang_mom = np.zeros(n_saved)
core_sep = np.zeros(n_saved)


def record(i, step):

    history[i] = pos

    times[i] = step * dt

    energy[i] = total_energy(pos, vel, mass)

    momentum[i] = np.linalg.norm(
        total_momentum(vel, mass)
    )

    ang_mom[i] = angular_momentum(
        pos, vel, mass
    )

    core_sep[i] = np.linalg.norm(
        pos[core_idx[0]] - pos[core_idx[1]]
    )


record(0, 0)

save_i = 1

for step in range(1, n_steps + 1):

    pos, vel = leapfrog_step(
        pos, vel, mass, dt
    )

    if step % SAVE_EVERY == 0:

        record(save_i, step)

        save_i += 1

    if step % 500 == 0:

        print(
            f"Simulation progress: "
            f"{step}/{n_steps}"
        )

print("\nSIMULATION COMPLETED!")


# ==========================================================
# 6. SCIENTIFIC DIAGNOSTICS
# ==========================================================

drift = (
    abs((energy[-1] - energy[0]) / energy[0]) * 100
    if energy[0] != 0 else 0
)

ang_drift = (
    abs((ang_mom[-1] - ang_mom[0]) / ang_mom[0]) * 100
    if ang_mom[0] != 0 else 0
)

final_pos = history[-1]

com = np.sum(
    mass[:, None] * final_pos,
    axis=0
) / mass.sum()

all_stars = np.concatenate([
    stars_1, stars_2
])

dist_com = np.sqrt(
    np.sum(
        (final_pos[all_stars] - com)**2,
        axis=1
    )
)

near = np.mean(dist_com < 20) * 100

closest = core_sep.min()

t_closest = times[core_sep.argmin()]

print("\n========== COLLISION REPORT ==========")

print(f"Relative velocity: {V_REL}")
print(f"Impact parameter: {IMPACT}")
print(f"Mass ratio: {MASS_RATIO}")

print(f"Energy drift: {drift:.4f}%")

print(f"Angular momentum drift: {ang_drift:.4f}%")

print(f"Maximum momentum: {momentum.max():.4f}")

print(f"Closest core approach: {closest:.3f}")

print(f"Time of closest approach: {t_closest:.3f}")

print(f"Final core separation: {core_sep[-1]:.3f}")

print(f"Stars near remnant: {near:.2f}%")

print("=======================================")


# ==========================================================
# 7. COSMIC BACKGROUND
# ==========================================================

rng = np.random.default_rng(42)

background_x = rng.uniform(-LIM, LIM, 180)
background_y = rng.uniform(-LIM, LIM, 180)

background_size = rng.uniform(1, 8, 180)


plt.rcParams.update({

    "figure.facecolor": BG,
    "axes.facecolor": BG,
    "axes.edgecolor": "#29314B",
    "axes.labelcolor": WHITE,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "text.color": WHITE,
    "font.family": "DejaVu Sans"

})


# ==========================================================
# 8. CINEMATIC SNAPSHOTS
# ==========================================================

frames = np.linspace(
    0, n_saved - 1, 6
).astype(int)

fig, axes = plt.subplots(
    2, 3,
    figsize=(15, 10)
)

fig.suptitle(
    "GALAXYLAB | THE COSMIC ENCOUNTER",
    fontsize=20,
    fontweight="bold",
    color=WHITE
)

for ax, f in zip(axes.ravel(), frames):

    ax.scatter(
        background_x,
        background_y,
        s=background_size,
        color="white",
        alpha=0.35
    )

    p1 = history[f, stars_1]
    p2 = history[f, stars_2]

    ax.scatter(
        p1[:, 0], p1[:, 1],
        s=45, color=BLUE, alpha=0.08
    )

    ax.scatter(
        p2[:, 0], p2[:, 1],
        s=45, color=GOLD, alpha=0.08
    )

    ax.scatter(
        p1[:, 0], p1[:, 1],
        s=9, color=BLUE
    )

    ax.scatter(
        p2[:, 0], p2[:, 1],
        s=9, color=GOLD
    )

    cx = history[f, core_idx, 0]
    cy = history[f, core_idx, 1]

    ax.scatter(
        cx, cy,
        s=300, color=PURPLE, alpha=0.15
    )

    ax.scatter(
        cx, cy,
        s=90, color=WHITE,
        edgecolors=PURPLE,
        linewidths=1.5
    )

    ax.set_xlim(-LIM, LIM)
    ax.set_ylim(-LIM, LIM)

    ax.set_aspect("equal")

    ax.set_title(
        f"TIME = {times[f]:.1f}",
        fontsize=11
    )

    ax.grid(alpha=0.10)

plt.tight_layout(rect=[0, 0, 1, 0.94])

plt.savefig(
    "collision_cinematic_snapshots.png",
    dpi=180,
    facecolor=BG
)

print("\nSaved cinematic snapshots!")


# ==========================================================
# 9. SCIENTIFIC DASHBOARD
# ==========================================================

fig2, axes2 = plt.subplots(
    1, 2,
    figsize=(14, 5)
)

fig2.suptitle(
    "GALAXYLAB | PHYSICS DIAGNOSTICS",
    fontsize=17,
    fontweight="bold"
)

axes2[0].plot(
    times,
    core_sep,
    color=BLUE,
    linewidth=2
)

axes2[0].fill_between(
    times,
    core_sep,
    color=BLUE,
    alpha=0.10
)

axes2[0].set_title("Galactic Core Separation")

axes2[0].set_xlabel("Simulation Time")

axes2[0].set_ylabel("Distance")

axes2[0].grid(alpha=0.15)


energy_change = (
    (energy - energy[0]) / abs(energy[0])
) * 100 if energy[0] != 0 else np.zeros_like(energy)

axes2[1].plot(
    times,
    energy_change,
    color=GOLD,
    linewidth=2
)

axes2[1].axhline(
    0,
    color=WHITE,
    alpha=0.3,
    linestyle="--"
)

axes2[1].set_title("Energy Conservation")

axes2[1].set_xlabel("Simulation Time")

axes2[1].set_ylabel("Energy Change (%)")

axes2[1].grid(alpha=0.15)

plt.tight_layout()

plt.savefig(
    "collision_physics_dashboard.png",
    dpi=180,
    facecolor=BG
)

print("Saved physics dashboard!")


# ==========================================================
# 10. CINEMATIC ANIMATION
# ==========================================================

fig3, ax = plt.subplots(
    figsize=(9, 9)
)

fig3.patch.set_facecolor(BG)

ax.set_facecolor(BG)

ax.set_xlim(-LIM, LIM)
ax.set_ylim(-LIM, LIM)

ax.set_aspect("equal")

ax.scatter(
    background_x,
    background_y,
    s=background_size,
    color="white",
    alpha=0.35
)

ax.grid(color="#68718F", alpha=0.10)

ax.set_title(
    "GALAXYLAB | GALACTIC MERGER",
    fontsize=16,
    fontweight="bold",
    pad=18
)

ax.set_xlabel("X POSITION")
ax.set_ylabel("Y POSITION")


# Particle glow

glow1 = ax.scatter(
    [], [], s=45, color=BLUE, alpha=0.12
)

glow2 = ax.scatter(
    [], [], s=45, color=GOLD, alpha=0.12
)


# Main particles

s1 = ax.scatter(
    [], [], s=9, color=BLUE
)

s2 = ax.scatter(
    [], [], s=9, color=GOLD
)


# Star trails

trail1 = ax.scatter(
    [], [], s=3, color=BLUE
)

trail2 = ax.scatter(
    [], [], s=3, color=GOLD
)


# Central cores

core_glow = ax.scatter(
    [], [], s=400, color=PURPLE, alpha=0.15
)

cores = ax.scatter(
    [], [],
    s=90,
    color=WHITE,
    edgecolors=PURPLE,
    linewidths=1.5,
    zorder=10
)


# Text

time_label = ax.text(
    0.04, 0.95, "",
    transform=ax.transAxes,
    fontsize=13,
    fontweight="bold",
    color=WHITE
)

info_label = ax.text(
    0.04, 0.04, "",
    transform=ax.transAxes,
    fontsize=9,
    color=MUTED
)


TRAIL_LENGTH = 25


def animate(f):

    p1 = history[f, stars_1]
    p2 = history[f, stars_2]

    glow1.set_offsets(p1)
    glow2.set_offsets(p2)

    s1.set_offsets(p1)
    s2.set_offsets(p2)

    # Recent particle history

    start = max(0, f - TRAIL_LENGTH)

    h1 = history[start:f+1, stars_1]
    h2 = history[start:f+1, stars_2]

    trail1.set_offsets(
        h1.reshape(-1, 2)
    )

    trail2.set_offsets(
        h2.reshape(-1, 2)
    )

    # Fading trail colours

    count = f - start + 1

    alphas = np.linspace(0.02, 0.40, count)

    colors1 = np.tile(
        to_rgba(BLUE),
        (count * N_STARS, 1)
    )

    colors2 = np.tile(
        to_rgba(GOLD),
        (count * N_STARS, 1)
    )

    colors1[:, 3] = np.repeat(
        alphas, N_STARS
    )

    colors2[:, 3] = np.repeat(
        alphas, N_STARS
    )

    trail1.set_facecolors(colors1)
    trail2.set_facecolors(colors2)

    # Galactic cores

    cx = history[f, core_idx, 0]
    cy = history[f, core_idx, 1]

    core_positions = np.column_stack((cx, cy))

    core_glow.set_offsets(core_positions)
    cores.set_offsets(core_positions)

    # Labels

    time_label.set_text(
        f"SIMULATION TIME | {times[f]:.2f}"
    )

    info_label.set_text(
        f"CORE DISTANCE: {core_sep[f]:.2f}  |  "
        f"ENERGY DRIFT: {energy_change[f]:.3f}%"
    )

    return (
        glow1, glow2,
        s1, s2,
        trail1, trail2,
        core_glow, cores,
        time_label, info_label
    )


ani = animation.FuncAnimation(
    fig3,
    animate,
    frames=n_saved,
    interval=35,
    blit=False,
    repeat=True
)


# ==========================================================
# 11. FINAL OUTPUT
# ==========================================================

print("\n====================================")
print("GALAXYLAB VISUALIZATION READY")
print("====================================")

print("1. collision_cinematic_snapshots.png")
print("2. collision_physics_dashboard.png")
print("3. Live galaxy collision animation")

print("\nEnjoy your cosmic encounter!")

plt.show() 