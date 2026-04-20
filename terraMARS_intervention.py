"""
TerraMARS Phase 5 — Intervention Comparison Engine
Compares different terraforming strategies to find optimal path
"""

import numpy as np
import matplotlib.pyplot as plt
import json

# ── Stage targets ────────────────────────────────────────────────────────────
STAGES = {
    0: {"name": "Current Mars",      "P_target": 0.6,  "T_target": -63, "bio_target": 0},
    1: {"name": "Pressure Buildup",  "P_target": 25,   "T_target": -40, "bio_target": 0},
    2: {"name": "Warming",           "P_target": 50,   "T_target": -10, "bio_target": 0},
    3: {"name": "Pioneer Biology",   "P_target": 80,   "T_target": 0,   "bio_target": 1e6},
    4: {"name": "Soil Formation",    "P_target": 100,  "T_target": 5,   "bio_target": 1e8},
    5: {"name": "Breathable",        "P_target": 101,  "T_target": 15,  "bio_target": 1e10},
}

# ── Scenarios to compare ─────────────────────────────────────────────────────
SCENARIOS = {
    "Conservative (CO2 only)": {
        "co2_release": {"start": 2050, "P_rate": 0.3, "T_rate": 0.15},
    },
    "Moderate (CO2 + Mirrors)": {
        "co2_release":     {"start": 2050, "P_rate": 0.3, "T_rate": 0.15},
        "orbital_mirrors": {"start": 2080, "T_rate": 0.15},
    },
    "Aggressive (All physical)": {
        "co2_release":     {"start": 2050, "P_rate": 0.5, "T_rate": 0.2},
        "orbital_mirrors": {"start": 2070, "T_rate": 0.2},
        "greenhouse_gas":  {"start": 2080, "T_rate": 0.15},
    },
    "Bio-focused": {
        "co2_release":     {"start": 2050, "P_rate": 0.3, "T_rate": 0.15},
        "cyanobacteria":   {"start": 2120, "bio_rate": 0.4},
        "engineered_bio":  {"start": 2180, "bio_rate": 0.6},
    },
    "Full Terraforming": {
        "co2_release":     {"start": 2050, "P_rate": 0.5, "T_rate": 0.2},
        "orbital_mirrors": {"start": 2070, "T_rate": 0.2},
        "greenhouse_gas":  {"start": 2080, "T_rate": 0.15},
        "cyanobacteria":   {"start": 2120, "bio_rate": 0.4},
        "engineered_bio":  {"start": 2180, "bio_rate": 0.6},
        "genetic_plants":  {"start": 2250, "bio_rate": 0.5},
    },
}

# ── Simulation params ────────────────────────────────────────────────────────
N_SIMULATIONS = 500
START_YEAR    = 2050
END_YEAR      = 2500
TIME_STEP     = 10

def simulate_trajectory(interventions, seed=None):
    """Run one trajectory with given interventions."""
    if seed is not None:
        np.random.seed(seed)

    P, T, bio = 0.6, -63.0, 0.0
    years = list(range(START_YEAR, END_YEAR + 1, TIME_STEP))
    history = {"year": [], "stage": []}

    for year in years:
        for name, params in interventions.items():
            if year >= params["start"]:
                if "P_rate" in params:
                    noise = np.random.normal(1.0, 0.3)
                    P += params["P_rate"] * TIME_STEP * max(0, noise)
                if "T_rate" in params:
                    noise = np.random.normal(1.0, 0.3)
                    T += params["T_rate"] * TIME_STEP * max(0, noise)
                if "bio_rate" in params and T > -20:
                    growth = params["bio_rate"] * TIME_STEP
                    bio = max(bio, 1.0) * (1 + growth * np.random.normal(1.0, 0.4))

        P   = min(P, 101)
        T   = min(T, 20)
        bio = min(bio, 1e11)

        stage = 0
        for s in range(5, -1, -1):
            if (P >= STAGES[s]["P_target"] * 0.9 and
                T >= STAGES[s]["T_target"] - 5 and
                bio >= STAGES[s]["bio_target"] * 0.5):
                stage = s
                break

        history["year"].append(year)
        history["stage"].append(stage)

    return history

# ── Run all scenarios ────────────────────────────────────────────────────────
print(f"Comparing {len(SCENARIOS)} terraforming scenarios...")
print(f"Running {N_SIMULATIONS} simulations per scenario\n")

scenario_results = {}
for scenario_name, interventions in SCENARIOS.items():
    print(f"Running: {scenario_name}")
    trajectories = []
    for i in range(N_SIMULATIONS):
        traj = simulate_trajectory(interventions, seed=i)
        trajectories.append(traj)
    scenario_results[scenario_name] = trajectories

# ── Analyze: time to each stage per scenario ─────────────────────────────────
print(f"\n{'='*70}")
print(f"TIME TO REACH EACH STAGE (50% probability)")
print(f"{'='*70}")
print(f"{'Scenario':<30} {'S1':>6} {'S2':>6} {'S3':>6} {'S4':>6} {'S5':>6}")
print(f"{'-'*70}")

years_common = scenario_results[list(SCENARIOS.keys())[0]][0]["year"]
scenario_timings = {}

for scenario_name, trajectories in scenario_results.items():
    timings = []
    for target_stage in range(1, 6):
        year_50 = None
        for t, year in enumerate(years_common):
            count = sum(1 for traj in trajectories if traj["stage"][t] >= target_stage)
            prob = count / N_SIMULATIONS
            if prob > 0.5 and year_50 is None:
                year_50 = year
                break
        timings.append(year_50 if year_50 else "---")

    scenario_timings[scenario_name] = timings
    row = f"{scenario_name:<30} "
    row += " ".join([f"{str(t):>6}" for t in timings])
    print(row)

# ── Find winner ──────────────────────────────────────────────────────────────
best_scenario = None
best_stage5_year = 9999
for name, timings in scenario_timings.items():
    if timings[4] != "---" and isinstance(timings[4], int):
        if timings[4] < best_stage5_year:
            best_stage5_year = timings[4]
            best_scenario = name

print(f"\n{'='*70}")
print(f"OPTIMAL STRATEGY: {best_scenario}")
print(f"Reaches breathable atmosphere by year: {best_stage5_year}")
print(f"{'='*70}")

# ── Save results ─────────────────────────────────────────────────────────────
results_out = {}
for name, timings in scenario_timings.items():
    results_out[name] = {
        "stage_1_year": timings[0] if timings[0] != "---" else None,
        "stage_2_year": timings[1] if timings[1] != "---" else None,
        "stage_3_year": timings[2] if timings[2] != "---" else None,
        "stage_4_year": timings[3] if timings[3] != "---" else None,
        "stage_5_year": timings[4] if timings[4] != "---" else None,
    }

with open("/home/exouser/jyotsna/terra_mars/intervention_results.json", "w") as f:
    json.dump({
        "best_scenario": best_scenario,
        "best_stage5_year": best_stage5_year,
        "scenarios": results_out
    }, f, indent=2)

print(f"\nResults saved to intervention_results.json")

# ── Plot comparison ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

# Stage probability over time for each scenario
for idx, (scenario_name, trajectories) in enumerate(scenario_results.items()):
    row = idx // 3
    col = idx % 3
    ax = axes[row, col]

    for target_stage in range(1, 6):
        probs = []
        for t in range(len(years_common)):
            count = sum(1 for traj in trajectories if traj["stage"][t] >= target_stage)
            probs.append(count / N_SIMULATIONS * 100)
        ax.plot(years_common, probs, label=f"Stage {target_stage}", linewidth=2,
                color=colors[target_stage-1])

    ax.set_title(scenario_name, fontsize=11, fontweight='bold')
    ax.set_xlabel("Year")
    ax.set_ylabel("Probability (%)")
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 105)

# Last panel: comparison of time to Stage 5
ax = axes[1, 2]
scenario_names = list(scenario_timings.keys())
stage5_years = []
for name in scenario_names:
    t = scenario_timings[name][4]
    stage5_years.append(t if isinstance(t, int) else END_YEAR + 50)

bars = ax.barh(range(len(scenario_names)),
                [y - 2050 for y in stage5_years],
                color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
ax.set_yticks(range(len(scenario_names)))
ax.set_yticklabels([n[:25] for n in scenario_names], fontsize=9)
ax.set_xlabel("Years from 2050 to Stage 5")
ax.set_title("Time to Breathable Atmosphere", fontsize=11, fontweight='bold')
ax.grid(alpha=0.3, axis='x')

for i, (bar, year) in enumerate(zip(bars, stage5_years)):
    if year > END_YEAR:
        ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2,
                " Not achieved", fontsize=9, va='center', color='red')
    else:
        ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2,
                f" {year}", fontsize=9, va='center')

plt.suptitle("TerraMARS Phase 5 — Intervention Scenario Comparison",
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("/home/exouser/jyotsna/terra_mars/intervention_comparison.png",
            dpi=150, bbox_inches="tight")
print(f"Plot saved to intervention_comparison.png")

print(f"\n{'='*70}")
print(f"PHASE 5 COMPLETE!")
print(f"{'='*70}")
