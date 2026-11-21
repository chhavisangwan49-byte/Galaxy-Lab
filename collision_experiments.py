"""
GALAXY LAB - Step 8: Collision Experiments (Experiment 3)
-------------------------------------------------------------
Same two-galaxy collision as collision.py, but run 4 times with ONE thing
changed each time, so we can see what each parameter does:

    A  Baseline          V_rel = 2.0, impact = 6,  mass ratio = 1.0
    B  Faster approach   V_rel = 3.5  (faster than the escape speed -> fly-by?)
    C  Wider miss        impact = 14  (galaxies barely touch)
    D  Unequal masses    mass ratio = 0.3  (galaxy 2 has only 30% of galaxy 1's central mass)

Outputs: a grid of snapshots, a comparison plot, and a table of numbers.
"""

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------
# 1. SETTINGS
# ---------------------------
N_STARS = 80             # stars PER galaxy (fewer than collision.py so 4 runs finish in reasonable time)
R_MIN = 1.0
GALAXY_RADIUS = 5.0
CENTRAL_MASS_1 = 50.0
STAR_MASS = 0.05
SEPARATION = 25.0

G = 1.0
eps = 0.1                # star-star softening
EPS_CORE = 1.0           # softening for the heavy central masses
dt = 0.01
n_steps = 5000
SAVE_EVERY = 25

# name, relative speed, impact parameter, mass ratio
SCENARIOS = [
    ("A: Baseline",        2.0,  6.0, 1.0),
    ("B: Faster approach", 3.5,  6.0, 1.0),
    ("C: Wider miss",      2.0, 14.0, 1.0),
    ("D: Unequal masses",  2.0,  6.0, 0.3),
]

# ---------------------------
# 2. BUILD A GALAXY
# ---------------------------
def make_galaxy(central_mass, spin, seed):
    rng = np.random.RandomState(seed)
    radii = R_MIN + (GALAXY_RADIUS - R_MIN) * np.sqrt(rng.uniform(0, 1, N_STARS))
    angles = rng.uniform(0, 2 * np.pi, N_STARS)
    order = np.argsort(radii)
    stars_inside = np.empty(N_STARS)
    stars_inside[order] = np.arange(N_STARS)
    v2 = G * central_mass * radii**2 / (radii**2 + EPS_CORE**2) ** 1.5 \
       + G * STAR_MASS * stars_inside * radii**2 / (radii**2 + eps**2) ** 1.5
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

# ---------------------------
# 3. RUN ONE COLLISION
# ---------------------------
def run_scenario(scenario):
    name, V_REL, IMPACT, RATIO = scenario

    pos1, vel1, mass1 = make_galaxy(CENTRAL_MASS_1, +1, seed=1)
    pos2, vel2, mass2 = make_galaxy(CENTRAL_MASS_1 * RATIO, +1, seed=2)
    M1, M2 = mass1.sum(), mass2.sum()

    # collision course in the centre-of-mass frame
    pos1 += np.array([-SEPARATION / 2, -IMPACT / 2])
    pos2 += np.array([+SEPARATION / 2, +IMPACT / 2])
    vel1 += np.array([+V_REL * M2 / (M1 + M2), 0.0])
    vel2 += np.array([-V_REL * M1 / (M1 + M2), 0.0])

    pos = np.vstack([pos1, pos2])
    vel = np.vstack([vel1, vel2])
    mass = np.concatenate([mass1, mass2])

    core = [0, N_STARS + 1]
    stars_1 = np.arange(1, N_STARS + 1)
    stars_2 = np.arange(N_STARS + 2, 2 * N_STARS + 2)

    soft = np.full(len(mass), eps)
    soft[core] = EPS_CORE
    EPS2 = np.maximum(soft[:, None], soft[None, :]) ** 2

    def acceleration(p):
        diff = p[None, :, :] - p[:, None, :]
        inv_d3 = (np.sum(diff**2, axis=2) + EPS2) ** -1.5
        np.fill_diagonal(inv_d3, 0.0)
        return G * np.sum(mass[None, :, None] * diff * inv_d3[:, :, None], axis=1)

    def energy(p, v):
        ke = np.sum(0.5 * mass * np.sum(v**2, axis=1))
        diff = p[None, :, :] - p[:, None, :]
        inv_r = 1.0 / np.sqrt(np.sum(diff**2, axis=2) + EPS2)
        np.fill_diagonal(inv_r, 0.0)
        return ke - 0.5 * G * np.sum(mass[:, None] * mass[None, :] * inv_r)

    n_saved = n_steps // SAVE_EVERY + 1
    history = np.zeros((n_saved, len(mass), 2))
    times = np.zeros(n_saved)
    core_sep = np.zeros(n_saved)

    E0 = energy(pos, vel)
    history[0] = pos
    core_sep[0] = np.linalg.norm(pos[core[0]] - pos[core[1]])
    acc = acceleration(pos)
    k = 1
    for step in range(1, n_steps + 1):
        vel_half = vel + acc * dt / 2
        pos = pos + vel_half * dt
        acc = acceleration(pos)
        vel = vel_half + acc * dt / 2
        if step % SAVE_EVERY == 0:
            history[k] = pos
            times[k] = step * dt
            core_sep[k] = np.linalg.norm(pos[core[0]] - pos[core[1]])
            k += 1
    E1 = energy(pos, vel)

    # ---- measurements ----
    # (1) Escaped stars: total energy > 0 in the centre-of-mass frame (kinetic + potential from ALL bodies)
    all_stars = np.concatenate([stars_1, stars_2])
    v_com = np.sum(mass[:, None] * vel, axis=0) / mass.sum()
    diff = pos[None, :, :] - pos[:, None, :]
    inv_r = 1.0 / np.sqrt(np.sum(diff**2, axis=2) + EPS2)
    np.fill_diagonal(inv_r, 0.0)
    potential = -G * np.sum(mass[None, :] * inv_r, axis=1)           # potential at each body
    kinetic = 0.5 * np.sum((vel - v_com) ** 2, axis=1)
    escaped = np.mean((kinetic + potential)[all_stars] > 0) * 100

    # (2) Are the two galaxies still bound to each other? (orbital energy of the two cores)
    m1, m2 = mass[core[0]], mass[core[1]]
    mu = m1 * m2 / (m1 + m2)
    d_vec = pos[core[0]] - pos[core[1]]
    v_vec = vel[core[0]] - vel[core[1]]
    e_pair = 0.5 * mu * np.sum(v_vec**2) - G * m1 * m2 / np.sqrt(np.sum(d_vec**2) + EPS_CORE**2)
    pair_bound = bool(e_pair < 0)

    return {
        "name": name, "V": V_REL, "b": IMPACT, "ratio": RATIO,
        "history": history, "times": times, "core_sep": core_sep,
        "core": core, "stars_1": stars_1, "stars_2": stars_2,
        "drift": abs((E1 - E0) / E0) * 100,
        "closest": core_sep.min(), "t_closest": times[core_sep.argmin()],
        "final_sep": core_sep[-1], "escaped": escaped, "pair_bound": pair_bound,
    }

# ---------------------------
# 4. TABLE + FIGURES
# ---------------------------
def make_outputs(results):
    print()
    print(f"{'Scenario':<20} | {'V_rel':>5} | {'b':>4} | {'ratio':>5} | {'Closest':>8} | {'Final sep':>9} | {'Galaxies bound?':>15} | {'Escaped stars':>13} | {'E drift':>8}")
    print("-" * 112)
    for r in results:
        print(f"{r['name']:<20} | {r['V']:>5.1f} | {r['b']:>4.0f} | {r['ratio']:>5.1f} | {r['closest']:>8.2f} | {r['final_sep']:>9.1f} | {('yes' if r['pair_bound'] else 'no'):>15} | {r['escaped']:>12.0f}% | {r['drift']:>7.3f}%")
    print("(Galaxies bound = the two cores are still gravitationally bound to each other; Escaped stars = stars with positive total energy)")

    # --- snapshot grid: one row per scenario, three moments each ---
    n_saved = len(results[0]["times"])
    cols = [0, n_saved // 2, n_saved - 1]
    fig, axes = plt.subplots(len(results), 3, figsize=(12, 4 * len(results)))
    for row, r in enumerate(results):
        # zoom out automatically if the galaxies fly far apart (so they never leave the picture)
        far = max(np.abs(r["history"][f, r["core"], :]).max() for f in cols)
        LIM = max(35, 1.2 * far + 8)
        for col, f in enumerate(cols):
            ax = axes[row, col]
            h = r["history"]
            ax.set_facecolor("black")
            ax.scatter(h[f, r["stars_1"], 0], h[f, r["stars_1"], 1], s=4, color="deepskyblue")
            ax.scatter(h[f, r["stars_2"], 0], h[f, r["stars_2"], 1], s=4, color="orange")
            ax.scatter(h[f, r["core"], 0], h[f, r["core"], 1], s=35, color="white", edgecolors="red")
            ax.set_xlim(-LIM, LIM); ax.set_ylim(-LIM, LIM); ax.set_aspect("equal")
            ax.set_title(f"{r['name']}   (t = {r['times'][f]:.0f})", fontsize=10)
    plt.tight_layout()
    plt.savefig("collision_comparison_snapshots.png", dpi=110)
    plt.close(fig)

    # --- comparison plots ---
    colors = ["tab:blue", "tab:red", "tab:green", "tab:purple"]
    fig2, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    for r, c in zip(results, colors):
        a1.plot(r["times"], r["core_sep"], color=c, label=r["name"])
    a1.set_xlabel("Time"); a1.set_ylabel("Distance between the two cores")
    a1.set_title("How the cores move in each collision"); a1.legend(); a1.grid(alpha=0.3)

    x = np.arange(len(results))
    esc = [r["escaped"] for r in results]
    bars = a2.bar(x, esc, color=colors)
    for b, r in zip(bars, results):
        a2.text(b.get_x() + b.get_width() / 2, b.get_height() + 1,
                "galaxies bound" if r["pair_bound"] else "galaxies unbound", ha="center", fontsize=8)
    a2.set_xticks(x); a2.set_xticklabels([r["name"] for r in results], fontsize=8)
    a2.set_ylabel("Stars that escaped the system (%)")
    a2.set_ylim(0, max(esc) * 1.3 + 5)
    a2.set_title("How many stars were thrown out?"); a2.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig("collision_comparison.png", dpi=130)
    plt.close(fig2)
    print("\nSaved collision_comparison_snapshots.png and collision_comparison.png")


if __name__ == "__main__":
    results = []
    for i, sc in enumerate(SCENARIOS, start=1):
        print(f"...running scenario {i}/{len(SCENARIOS)}: {sc[0]} (this takes a few minutes)...")
        results.append(run_scenario(sc))
    make_outputs(results)