"""
TerraMARS Phase 4 — Monte Carlo Stage Transition Model
Simulates probabilistic progression through 6 terraforming stages
"""

import numpy as np
import matplotlib.pyplot as plt
import json

# ── Stage definitions ────────────────────────────────────────────────────────
STAGES = {
    0: {"name": "Current Mars",        "P_target": 0.6,   "T_target": -63, "bio_target": 0},
    1: {"name": "Pressure Buildup",    "P_target": 25,    "T_target": -40, "bio_target": 0},
    2: {"name": "Warming Phase",       "P_target": 50,    "T_target": -10, "bio_target": 0},
    3: {"name": "Pioneer Biology",     "P_target": 80,    "T_target": 0,   "bio_target": 1e6},
    4: {"name": "Soil Formation",      "P_target": 100,   "T_target": 5,   "bio_target": 1e8},
    5: {"name": "Breathable Atmos.",   "P_target": 101,   "T_target": 15,  "bio_target": 1e10},
}

# ── Simulation parameters ────────────────────────────────────────────────────
N_SIMULATIONS = 1000
START_YEAR    = 2050
END_YEAR      = 2500
TIME_STEP     = 10  # years

# ── Interventions (can be toggled) ───────────────────────────────────────────
INTERVENTIONS = {
    "co2_release":     {"start": 2050, "P_rate": 0.5,   "T_rate": 0.2},
    "orbital_mirrors": {"start": 2080, "P_rate": 0.0,   "T_rate": 0.15},
    "cyanobacteria":   {"start": 2150, "bio_rate": 0.3},
    "engineered_bio":  {"start": 2200, "bio_rate": 0.5},
    "genetic_plants":  {"start": 2300, "bio_rate": 0.4},
}

def simulate_trajectory(seed=None):
    """Simulate one terraforming trajectory with noise."""
    if seed is not None:
        np.random.seed(seed)

    # Initial conditions (current Mars)
    P   = 0.6
    T   = -63.0
    bio = 0.0

    # Track values over time
    years = list(range(START_YEAR, END_YEAR + 1, TIME_STEP))
    history = {"year": [], "P": [], "T": [], "bio": [], "stage": []}

    for year in years:
        # Apply active interventions with uncertainty
        for name, params in INTERVENTIONS.items():
            if year >= params["start"]:
                if "P_rate" in params:
                    noise = np.random.normal(1.0, 0.3)  # 30% uncertainty
                    P += params["P_rate"] * TIME_STEP * max(0, noise)
                if "T_rate" in params:
                    noise = np.random.normal(1.0, 0.3)
                    T += params["T_rate"] * TIME_STEP * max(0, noise)
                if "bio_rate" in params and T > -20:  # biology needs warmth
                    growth = params["bio_rate"] * TIME_STEP
                    bio = max(bio, 1.0) * (1 + growth * np.random.normal(1.0, 0.4))

        # Cap values at reasonable maximums
        P   = min(P, 101)
        T   = min(T, 20)
        bio = min(bio, 1e11)

        # Determine current stage
        stage = 0
        for s in range(5, -1, -1):
            if (P >= STAGES[s]["P_target"] * 0.9 and
                T >= STAGES[s]["T_target"] - 5 and
                bio >= STAGES[s]["bio_target"] * 0.5):
                stage = s
                break

        history["year"].append(year)
        history["P"].append(P)
        history["T"].append(T)
        history["bio"].append(bio)
        history["stage"].append(stage)

    return history

# ── Run Monte Carlo ──────────────────────────────────────────────────────────
print(f"Running {N_SIMULATIONS} Monte Carlo simulations...")
print(f"Time range: {START_YEAR} → {END_YEAR}")
print(f"Interventions:")
for name, params in INTERVENTIONS.items():
    print(f"  {name:20s} starting year {params['start']}")
print()

all_trajectories = []
for i in range(N_SIMULATIONS):
    trajectory = simulate_trajectory(seed=i)
    all_trajectories.append(trajectory)
    if (i + 1) % 100 == 0:
        print(f"  {i+1}/{N_SIMULATIONS} simulations complete")

# ── Analyze results ──────────────────────────────────────────────────────────
years = all_trajectories[0]["year"]
n_years = len(years)

# For each year, calculate probability of being in each stage
stage_probs = np.zeros((6, n_years))
for traj in all_trajectories:
    for t, stage in enumerate(traj["stage"]):
        stage_probs[stage, t] += 1
stage_probs /= N_SIMULATIONS

# Calculate when each stage becomes "likely" (>50%)
print(f"\n{'='*60}")
print(f"STAGE ACHIEVEMENT PROBABILITIES")
print(f"{'='*60}")
print(f"{'Stage':<25} {'Year':>6} {'Probability':>12}")
print(f"{'-'*60}")

for target_stage in range(1, 6):
    # Probability of being >= target stage
    prob_achieved = np.zeros(n_years)
    for t in range(n_years):
        count = sum(1 for traj in all_trajectories if traj["stage"][t] >= target_stage)
        prob_achieved[t] = count / N_SIMULATIONS

    # Find first year where probability > 50%
    year_50 = None
    for t, year in enumerate(years):
        if prob_achieved[t] > 0.5:
            year_50 = year
            break

    max_prob = prob_achieved.max()
    year_max = years[np.argmax(prob_achieved)]

    if year_50:
        print(f"Stage {target_stage} ({STAGES[target_stage]['name']:<18}) {year_50:>6}  50% by then")
    else:
        print(f"Stage {target_stage} ({STAGES[target_stage]['name']:<18}) {'---':>6}  max {max_prob*100:.1f}% at {year_max}")

# ── Save results to JSON ─────────────────────────────────────────────────────
results = {
    "n_simulations": N_SIMULATIONS,
    "start_year": START_YEAR,
    "end_year": END_YEAR,
    "time_step": TIME_STEP,
    "interventions": INTERVENTIONS,
    "years": years,
    "stage_probabilities": {
        f"stage_{s}": stage_probs[s].tolist() for s in range(6)
    },
    "mean_trajectory": {
        "P":   np.mean([t["P"]   for t in all_trajectories], axis=0).tolist(),
        "T":   np.mean([t["T"]   for t in all_trajectories], axis=0).tolist(),
        "bio": np.mean([t["bio"] for t in all_trajectories], axis=0).tolist(),
    }
}

with open("/home/exouser/jyotsna/terra_mars/montecarlo_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to montecarlo_results.json")

# ── Generate plots ───────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Stage probability over time
ax = axes[0, 0]
for s in range(6):
    ax.plot(years, stage_probs[s]*100, label=f"Stage {s}: {STAGES[s]['name']}", linewidth=2)
ax.set_xlabel("Year")
ax.set_ylabel("Probability (%)")
ax.set_title("Stage Probability Over Time")
ax.legend(fontsize=8, loc='upper right')
ax.grid(alpha=0.3)

# Plot 2: Pressure trajectory
ax = axes[0, 1]
P_mean = np.mean([t["P"] for t in all_trajectories], axis=0)
P_std  = np.std( [t["P"] for t in all_trajectories], axis=0)
ax.plot(years, P_mean, color="blue", linewidth=2, label="Mean pressure")
ax.fill_between(years, P_mean - P_std, P_mean + P_std, alpha=0.2, color="blue", label="±1σ uncertainty")
ax.axhline(y=25,  color="orange", linestyle="--", label="Stage 1 target")
ax.axhline(y=50,  color="red",    linestyle="--", label="Stage 2 target")
ax.axhline(y=101, color="green",  linestyle="--", label="Earth-like")
ax.set_xlabel("Year")
ax.set_ylabel("Pressure (kPa)")
ax.set_title("Atmospheric Pressure Trajectory")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# Plot 3: Temperature trajectory
ax = axes[1, 0]
T_mean = np.mean([t["T"] for t in all_trajectories], axis=0)
T_std  = np.std( [t["T"] for t in all_trajectories], axis=0)
ax.plot(years, T_mean, color="red", linewidth=2, label="Mean temperature")
ax.fill_between(years, T_mean - T_std, T_mean + T_std, alpha=0.2, color="red", label="±1σ uncertainty")
ax.axhline(y=-10, color="orange", linestyle="--", label="Stage 2 target")
ax.axhline(y=0,   color="green",  linestyle="--", label="Freezing point")
ax.set_xlabel("Year")
ax.set_ylabel("Temperature (°C)")
ax.set_title("Surface Temperature Trajectory")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# Plot 4: Biology density
ax = axes[1, 1]
bio_mean = np.mean([t["bio"] for t in all_trajectories], axis=0)
ax.semilogy(years, np.maximum(bio_mean, 1), color="green", linewidth=2, label="Cell density")
ax.axhline(y=1e6,  color="orange", linestyle="--", label="Stage 3 target")
ax.axhline(y=1e8,  color="red",    linestyle="--", label="Stage 4 target")
ax.axhline(y=1e10, color="blue",   linestyle="--", label="Stage 5 target")
ax.set_xlabel("Year")
ax.set_ylabel("Cells per gram regolith (log scale)")
ax.set_title("Biological Colonization")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

plt.suptitle("TerraMARS Phase 4 — Monte Carlo Stage Transition Simulation",
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("/home/exouser/jyotsna/terra_mars/montecarlo_plots.png", dpi=150, bbox_inches="tight")
print(f"Plots saved to montecarlo_plots.png")

print(f"\n{'='*60}")
print(f"PHASE 4 COMPLETE!")
print(f"{'='*60}")
