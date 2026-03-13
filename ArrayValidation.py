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

# ---------------------------------------------------------------------------
# Re-import everything from the training script so classes match pickle
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(__file__))
from ArrayBasedTraining import (
    Vehicle, TimingBlock, VTYPES,
    INTERSECTIONS_H, INTERSECTIONS_V, SEGMENT_LENGTHS_H, SEGMENT_LENGTHS_V,
    MAP_END, GRID_SIZE, SPAWN_RATE, SPAWNS_PER_SECOND, SPEED_LIMIT,
    build_road_arrays, pos_to_seg,
    _dist_ahead_in_arrays, _dist_to_stop_line,
)

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
PKL_PATH = os.path.join(os.path.dirname(__file__), 'best_timing_array.pkl')

with open(PKL_PATH, 'rb') as f:
    best_blocks = pickle.load(f)

print(f"Loaded {len(best_blocks)} TimingBlocks from {PKL_PATH}")

# ---------------------------------------------------------------------------
# Flatten chromosome into a 3600-second schedule
# ---------------------------------------------------------------------------
flat = []
for b in best_blocks:
    flat.extend(b.seq)
if len(flat) < 3600:
    flat.extend([4] * (3600 - len(flat)))
schedule = np.array(flat[:3600], dtype=int)

# ---------------------------------------------------------------------------
# Also build a baseline schedule for comparison
# ---------------------------------------------------------------------------
baseline_flat = []
for _ in range(60):
    baseline_flat.extend(TimingBlock(30, 30).seq)
if len(baseline_flat) < 3600:
    baseline_flat.extend([4] * (3600 - len(baseline_flat)))
baseline_schedule = np.array(baseline_flat[:3600], dtype=int)

# ---------------------------------------------------------------------------
# Plot 1 — Evolved phase sequence (first 20 minutes)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(schedule[:1200], color='#1f77b4', drawstyle='steps-post', linewidth=1.5,
        label='Evolved')
ax.plot(baseline_schedule[:1200], color='#d62728', drawstyle='steps-post',
        linewidth=1.0, alpha=0.5, linestyle='--', label='Baseline (30s/30s)')
ax.set_title("Evolved vs Baseline Phase Sequence (First 20 Minutes)", fontsize=14)
ax.set_xlabel("Simulation Time (seconds)", fontsize=12)
ax.set_ylabel("Light State", fontsize=12)
ax.set_yticks([0, 1, 2, 3, 4])
ax.set_yticklabels(['0: NS Green', '1: NS Yellow', '2: EW Green',
                    '3: EW Yellow', '4: All Red'])
ax.grid(True, axis='x', alpha=0.3)
ax.legend()
plt.tight_layout()
plt.savefig('plot_phase_sequence.png', dpi=150)
plt.show()
print("Saved: plot_phase_sequence.png")

# ---------------------------------------------------------------------------
# Full simulation replay — collect per-vehicle stats + density history
# ---------------------------------------------------------------------------
def replay(sched, seed=3000, label=""):
    random.seed(seed)
    Vehicle._next_id = 1
    road_arrays = build_road_arrays()

    vehicles = []
    active_vehicles = []
    density_history = []

    flat_sched = np.array(sched, dtype=int)

    for t in range(3600):
        for _ in range(SPAWNS_PER_SECOND):
            if random.random() < SPAWN_RATE:
                axis      = random.choice(['h', 'v'])
                direction = random.choice([1, -1])
                ch_idx    = random.randint(0, GRID_SIZE - 1)
                vtype     = random.choice(['car', 'bus', 'truck'])
                v_len     = VTYPES[vtype]['length']

                spawn_abs = 0.0 if direction == 1 else float(MAP_END)
                si, li    = pos_to_seg(spawn_abs, axis)
                key       = (axis, ch_idx, si, direction)
                arr       = road_arrays.get(key)
                can_spawn = False
                if arr is not None:
                    check = arr[:v_len + SPEED_LIMIT + 2] if direction == 1 \
                            else arr[-(v_len + SPEED_LIMIT + 2):]
                    can_spawn = not np.any(check != 0)

                if can_spawn:
                    nv = Vehicle(vtype, axis, direction, ch_idx)
                    nv.stamp(road_arrays)
                    vehicles.append(nv)
                    active_vehicles.append(nv)

        light = int(flat_sched[t])
        next_active = []
        for v in active_vehicles:
            dist = min(_dist_ahead_in_arrays(v, road_arrays),
                       _dist_to_stop_line(v, light))
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

    for vt in ('car', 'bus', 'truck'):
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
ev_vehicles, ev_finished, ev_density, ev_score = replay(schedule,        seed=3000, label="EVOLVED STRATEGY")

print("\nReplaying BASELINE strategy (30s NS / 30s EW)...")
bl_vehicles, bl_finished, bl_density, bl_score = replay(baseline_schedule, seed=3000, label="BASELINE STRATEGY")

# ---------------------------------------------------------------------------
# Plot 2 — Per-vehicle-type avg travel time (evolved vs baseline)
# ---------------------------------------------------------------------------
vtypes = ['car', 'bus', 'truck']
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
plt.savefig('plot_vehicle_breakdown.png', dpi=150)
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
plt.savefig('plot_finish_rate.png', dpi=150)
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
plt.savefig('plot_density.png', dpi=150)
plt.show()
print("Saved: plot_density.png")

print(f"\nFinal fitness — Evolved: {ev_score:.2f}  |  Baseline: {bl_score:.2f}  "
      f"|  Improvement: {bl_score - ev_score:+.2f}")
