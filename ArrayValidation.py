"""
ArrayValidation.py
------------------
Loads best_timing_array.pkl produced by ArrayBasedTraining.py and produces:
  1. A phase-sequence plot of the evolved chromosome (first 20 minutes)
  2. A per-vehicle-type breakdown of avg travel time and avg idling time
  3. A finish-rate bar chart (cars / buses / trucks)
  4. An active-vehicle density plot over the full hour with trendline
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import random
import sys
import os


def _pkl_from_argv():
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == '--pkl' and i + 1 < len(args):
            return args[i + 1]
        i += 1
    return None


# ---------------------------------------------------------------------------
# Re-import everything from the training script so classes match pickle
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(__file__))
from ArrayBasedTraining import (
    Vehicle,
    TimingBlock,
    IntersectionTiming,
    VTYPES,
    INTERSECTIONS_H,
    INTERSECTIONS_V,
    SEGMENT_LENGTHS_H,
    SEGMENT_LENGTHS_V,
    MAP_END,
    GRID_SIZE,
    N_INTERSECTIONS,
    SPAWN_RATE,
    SPAWNS_PER_SECOND,
    SPEED_LIMIT,
    build_road_arrays,
    build_schedules,
    build_baseline_genes,
    _dist_ahead_in_arrays,
    _dist_to_stop_line,
    _sort_vehicles_for_step,
    try_spawn_one_vehicle,
    resolve_pkl_path,
)

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
PKL_PATH = os.path.abspath(resolve_pkl_path(_pkl_from_argv()))
OUTPUT_DIR = os.path.dirname(PKL_PATH)

with open(PKL_PATH, 'rb') as f:
    best_blocks = pickle.load(f)

print(f"Loaded {len(best_blocks)} entries from {PKL_PATH}")
print(f"Saving plots to: {OUTPUT_DIR}")

# ---------------------------------------------------------------------------
# Build per-intersection schedules
# ---------------------------------------------------------------------------
ev_schedules = build_schedules(best_blocks)
bl_schedules = build_schedules(build_baseline_genes())

# ---------------------------------------------------------------------------
# Plot 1 — Phase-sequence heatmap (all intersections, first 20 minutes)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(GRID_SIZE, GRID_SIZE, figsize=(16, 10), sharex=True, sharey=True)
from matplotlib.colors import ListedColormap
phase_cmap = ListedColormap(['#2ca02c', '#ffdc00', '#1f77b4', '#ff851b', '#d62728'])
phase_labels = ['NS Grn', 'NS Yel', 'EW Grn', 'EW Yel', 'AllRed']

for row in range(GRID_SIZE):
    for col in range(GRID_SIZE):
        ix_id = row * GRID_SIZE + col
        ax = axes[row][col]
        ev_s = ev_schedules[ix_id][:1200]
        bl_s = bl_schedules[ix_id][:1200]
        ax.imshow(np.vstack([ev_s, bl_s]), aspect='auto', cmap=phase_cmap,
                  vmin=0, vmax=4, interpolation='nearest',
                  extent=[0, 1200, 2, 0])
        ax.set_yticks([0.5, 1.5])
        ax.set_yticklabels(['Evo', 'Base'], fontsize=8)
        t = best_blocks[ix_id]
        ax.set_title(f"({row},{col}) g={t.g_ns}/{t.g_ew} off={t.offset}", fontsize=9)

fig.suptitle("Per-Intersection Phase Sequences (first 20 min)", fontsize=14)
fig.supxlabel("Simulation Time (seconds)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'plot_phase_sequence.png'), dpi=150)
plt.show()
print("Saved: plot_phase_sequence.png")

# ---------------------------------------------------------------------------
# Full simulation replay — collect per-vehicle stats + density history
# ---------------------------------------------------------------------------
def replay(schedules, seed=3000, label=""):
    random.seed(seed)
    Vehicle._next_id = 1
    road_arrays = build_road_arrays()

    vehicles = []
    active_vehicles = []
    density_history = []

    for t in range(3600):
        for _ in range(SPAWNS_PER_SECOND):
            if random.random() < SPAWN_RATE:
                try_spawn_one_vehicle(road_arrays, vehicles, active_vehicles)

        _sort_vehicles_for_step(active_vehicles)
        next_active = []
        for v in active_vehicles:
            dist = min(_dist_ahead_in_arrays(v, road_arrays),
                       _dist_to_stop_line(v, schedules, t))
            v.step(road_arrays, dist)
            if not v.finished:
                next_active.append(v)
        active_vehicles = next_active
        density_history.append(len(active_vehicles))

    finished   = [v for v in vehicles if v.finished]
    unfinished = [v for v in vehicles if not v.finished]
    score      = sum(v.travel_time + v.idling_time * 2 for v in vehicles)
    normalised = score / max(len(vehicles), 1)

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  Total spawned : {len(vehicles)}")
    print(f"  Finished      : {len(finished)}  ({100*len(finished)/max(len(vehicles),1):.1f}%)")
    print(f"  Unfinished    : {len(unfinished)}")
    print(f"  Fitness score : {normalised:.2f}")
    print(f"{'='*60}")

    for vt in ('car', 'bus', 'truck', 'goal_car'):
        vt_all  = [v for v in vehicles  if v.vtype == vt]
        vt_fin  = [v for v in finished  if v.vtype == vt]
        if not vt_all:
            continue
        avg_tt  = sum(v.travel_time  for v in vt_fin) / max(len(vt_fin), 1)
        avg_idl = sum(v.idling_time  for v in vt_all) / len(vt_all)
        fin_pct = 100 * len(vt_fin) / len(vt_all)
        print(f"  {vt:5s} | spawned={len(vt_all):3d} | finished={len(vt_fin):3d} ({fin_pct:4.1f}%) "
              f"| avg travel={avg_tt:6.1f}s | avg idle={avg_idl:5.1f}s")

    return vehicles, finished, density_history, normalised


print("\nReplaying EVOLVED strategy...")
ev_vehicles, ev_finished, ev_density, ev_score = replay(ev_schedules,  seed=3000, label="EVOLVED STRATEGY")

print("\nReplaying BASELINE strategy (30s NS / 30s EW, no offsets)...")
bl_vehicles, bl_finished, bl_density, bl_score = replay(bl_schedules, seed=3000, label="BASELINE STRATEGY")

# ---------------------------------------------------------------------------
# Plot 2 — Per-vehicle-type avg travel time (evolved vs baseline)
# ---------------------------------------------------------------------------
vtypes = ['car', 'bus', 'truck', 'goal_car']
x = np.arange(len(vtypes))
width = 0.35

def avg_tt_by_type(finished, vt):
    vals = [v.travel_time for v in finished if v.vtype == vt]
    return np.mean(vals) if vals else 0

def avg_idl_by_type(vehicles, vt):
    vals = [v.idling_time for v in vehicles if v.vtype == vt]
    return np.mean(vals) if vals else 0

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Travel time
ev_tt  = [avg_tt_by_type(ev_finished, vt) for vt in vtypes]
bl_tt  = [avg_tt_by_type(bl_finished, vt) for vt in vtypes]
axes[0].bar(x - width/2, ev_tt, width, label='Evolved',   color='#1f77b4')
axes[0].bar(x + width/2, bl_tt, width, label='Baseline',  color='#d62728', alpha=0.7)
axes[0].set_title("Avg Travel Time by Vehicle Type (finished only)")
axes[0].set_ylabel("Seconds")
axes[0].set_xticks(x)
axes[0].set_xticklabels(vtypes)
axes[0].legend()
axes[0].grid(axis='y', alpha=0.3)

# Idling time
ev_idl = [avg_idl_by_type(ev_vehicles, vt) for vt in vtypes]
bl_idl = [avg_idl_by_type(bl_vehicles, vt) for vt in vtypes]
axes[1].bar(x - width/2, ev_idl, width, label='Evolved',  color='#1f77b4')
axes[1].bar(x + width/2, bl_idl, width, label='Baseline', color='#d62728', alpha=0.7)
axes[1].set_title("Avg Idling Time by Vehicle Type (all spawned)")
axes[1].set_ylabel("Seconds")
axes[1].set_xticks(x)
axes[1].set_xticklabels(vtypes)
axes[1].legend()
axes[1].grid(axis='y', alpha=0.3)

plt.suptitle("Evolved vs Baseline: Travel & Idling Breakdown", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'plot_vehicle_breakdown.png'), dpi=150)
plt.show()
print("Saved: plot_vehicle_breakdown.png")

# ---------------------------------------------------------------------------
# Plot 3 — Finish rate by vehicle type
# ---------------------------------------------------------------------------
def finish_rate(vehicles, vt):
    total  = [v for v in vehicles if v.vtype == vt]
    fin    = [v for v in total    if v.finished]
    return 100 * len(fin) / max(len(total), 1)

ev_fr = [finish_rate(ev_vehicles, vt) for vt in vtypes]
bl_fr = [finish_rate(bl_vehicles, vt) for vt in vtypes]

fig, ax = plt.subplots(figsize=(7, 5))
ax.bar(x - width/2, ev_fr, width, label='Evolved',  color='#1f77b4')
ax.bar(x + width/2, bl_fr, width, label='Baseline', color='#d62728', alpha=0.7)
ax.set_title("Finish Rate by Vehicle Type (%)", fontsize=13)
ax.set_ylabel("% of spawned vehicles that finished")
ax.set_xticks(x)
ax.set_xticklabels(vtypes)
ax.set_ylim(0, 110)
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'plot_finish_rate.png'), dpi=150)
plt.show()
print("Saved: plot_finish_rate.png")

# ---------------------------------------------------------------------------
# Plot 4 — Active vehicle density over time with trendline
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 5))
t_axis = np.arange(3600)

ax.plot(ev_density, color='#1f77b4', linewidth=1.5, label='Evolved', alpha=0.85)
ax.plot(bl_density, color='#d62728', linewidth=1.5, label='Baseline', alpha=0.85)

# Trendlines
for data, col in [(ev_density, '#1f77b4'), (bl_density, '#d62728')]:
    z = np.polyfit(t_axis, data, 1)
    ax.plot(t_axis, np.poly1d(z)(t_axis), color=col, linestyle='--', linewidth=1.2, alpha=0.6)

ax.set_title("Gridlock Analysis: Active Vehicles Over Time", fontsize=14)
ax.set_xlabel("Simulation Time (seconds)", fontsize=12)
ax.set_ylabel("Active vehicles on grid", fontsize=12)
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'plot_density.png'), dpi=150)
plt.show()
print("Saved: plot_density.png")

print(f"\nFinal fitness — Evolved: {ev_score:.2f}  |  Baseline: {bl_score:.2f}  "
      f"|  Improvement: {bl_score - ev_score:+.2f}")
