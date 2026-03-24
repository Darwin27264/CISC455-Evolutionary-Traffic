"""
ArrayValidation.py
------------------
Loads best_timing_array.pkl produced by ArrayBasedTraining.py and produces:
  1. A phase-sequence plot of the evolved chromosome (first 20 minutes)
  2. A per-vehicle-type breakdown of avg travel time and avg idling time
  3. A finish-rate bar chart (cars / buses / trucks)
  4. An active-vehicle density plot over the full hour with trendline
  5. Scatter plots: mean adjacent road-segment length vs g_ns, g_ew, and phase offset
     (Pearson r); exploratory — small grids yield noisy correlations.
    6. Directional signal synchronization analysis (global phase alignment +
         travel-time-shifted green-wave progression) vs baseline.

Display: phase sequence opens in its own window; other plots are grouped into one
scalable dashboard window. Each plot is still saved as a separate PNG file.
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
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
# Plot helpers (draw on supplied axes so we can save separate PNGs + one dashboard)
# ---------------------------------------------------------------------------

PHASE_CMAP = ListedColormap(['#2ca02c', '#ffdc00', '#1f77b4', '#ff851b', '#d62728'])


def draw_phase_heatmap(fig, axes, ev_schedules, bl_schedules, best_blocks):
    """Populate GRID_SIZE x GRID_SIZE axes with phase imshow."""
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            ix_id = row * GRID_SIZE + col
            ax = axes[row][col]
            ev_s = ev_schedules[ix_id][:1200]
            bl_s = bl_schedules[ix_id][:1200]
            ax.imshow(
                np.vstack([ev_s, bl_s]),
                aspect='auto',
                cmap=PHASE_CMAP,
                vmin=0,
                vmax=4,
                interpolation='nearest',
                extent=[0, 1200, 2, 0],
            )
            ax.set_yticks([0.5, 1.5])
            ax.set_yticklabels(['Evo', 'Base'], fontsize=8)
            t = best_blocks[ix_id]
            ax.set_title(f"({row},{col}) g={t.g_ns}/{t.g_ew} off={t.offset}", fontsize=9)
    fig.suptitle("Per-Intersection Phase Sequences (first 20 min)", fontsize=14)
    fig.supxlabel("Simulation Time (seconds)")


def draw_vehicle_breakdown(ax_tt, ax_idl, vtypes, x, width, ev_tt, bl_tt, ev_idl, bl_idl):
    ax_tt.bar(x - width / 2, ev_tt, width, label='Evolved', color='#1f77b4')
    ax_tt.bar(x + width / 2, bl_tt, width, label='Baseline', color='#d62728', alpha=0.7)
    ax_tt.set_title("Avg travel time (finished only)")
    ax_tt.set_ylabel("Seconds")
    ax_tt.set_xticks(x)
    ax_tt.set_xticklabels(vtypes)
    ax_tt.legend()
    ax_tt.grid(axis='y', alpha=0.3)
    ax_tt.tick_params(axis='x', labelrotation=0)

    ax_idl.bar(x - width / 2, ev_idl, width, label='Evolved', color='#1f77b4')
    ax_idl.bar(x + width / 2, bl_idl, width, label='Baseline', color='#d62728', alpha=0.7)
    ax_idl.set_title("Avg idling time (all spawned)")
    ax_idl.set_ylabel("Seconds")
    ax_idl.set_xticks(x)
    ax_idl.set_xticklabels(vtypes)
    ax_idl.legend()
    ax_idl.grid(axis='y', alpha=0.3)


def draw_finish_rate(ax, vtypes, x, width, ev_fr, bl_fr):
    ax.bar(x - width / 2, ev_fr, width, label='Evolved', color='#1f77b4')
    ax.bar(x + width / 2, bl_fr, width, label='Baseline', color='#d62728', alpha=0.7)
    ax.set_title("Finish rate by vehicle type (%)")
    ax.set_ylabel("% of spawned vehicles that finished")
    ax.set_xticks(x)
    ax.set_xticklabels(vtypes)
    ax.set_ylim(0, 110)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)


def draw_density(ax, t_axis, ev_density, bl_density):
    ax.plot(ev_density, color='#1f77b4', linewidth=1.5, label='Evolved', alpha=0.85)
    ax.plot(bl_density, color='#d62728', linewidth=1.5, label='Baseline', alpha=0.85)
    for data, col in [(ev_density, '#1f77b4'), (bl_density, '#d62728')]:
        z = np.polyfit(t_axis, data, 1)
        ax.plot(t_axis, np.poly1d(z)(t_axis), color=col, linestyle='--', linewidth=1.2, alpha=0.6)
    ax.set_title("Active vehicles over time (grid occupancy)")
    ax.set_xlabel("Simulation time (s)")
    ax.set_ylabel("Active vehicles on grid")
    ax.legend()
    ax.grid(alpha=0.3)


def draw_correlation_scatters(axes3, seg_mean, y_ns, y_ew, y_off, r_ns, r_ew, r_off, scatter_kw):
    axes3[0].scatter(seg_mean, y_ns, c="#2ca02c", **scatter_kw)
    axes3[0].set_xlabel("Mean adjacent segment length (m)")
    axes3[0].set_ylabel("NS green g_ns (s)")
    axes3[0].set_title(f"NS green vs road length\nr = {r_ns:+.3f}")
    axes3[0].grid(alpha=0.3)

    axes3[1].scatter(seg_mean, y_ew, c="#1f77b4", **scatter_kw)
    axes3[1].set_xlabel("Mean adjacent segment length (m)")
    axes3[1].set_ylabel("EW green g_ew (s)")
    axes3[1].set_title(f"EW green vs road length\nr = {r_ew:+.3f}")
    axes3[1].grid(alpha=0.3)

    axes3[2].scatter(seg_mean, y_off, c="#ff7f0e", **scatter_kw)
    axes3[2].set_xlabel("Mean adjacent segment length (m)")
    axes3[2].set_ylabel("Phase offset (s)")
    axes3[2].set_title(f"Offset vs road length\nr = {r_off:+.3f}")
    axes3[2].grid(alpha=0.3)


def draw_signal_sync_comparison(ax, labels, ev_vals, bl_vals, width=0.36):
    idx = np.arange(len(labels))
    ax.bar(idx - width / 2, ev_vals, width, label='Evolved', color='#1f77b4')
    ax.bar(idx + width / 2, bl_vals, width, label='Baseline', color='#d62728', alpha=0.7)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score (0..1)")
    ax.set_title("Signal synchronization comparison")
    ax.set_xticks(idx)
    ax.set_xticklabels(labels, rotation=12, ha='right')
    ax.grid(axis='y', alpha=0.3)
    ax.legend()

    for i, (ev, bl) in enumerate(zip(ev_vals, bl_vals)):
        ax.text(i - width / 2, min(1.02, ev + 0.02), f"{ev:.3f}", ha='center', va='bottom', fontsize=8)
        ax.text(i + width / 2, min(1.02, bl + 0.02), f"{bl:.3f}", ha='center', va='bottom', fontsize=8)


def _savefig_only(path, dpi=150):
    plt.savefig(path, dpi=dpi, bbox_inches='tight')


def _direction_state_ns(phases: np.ndarray) -> np.ndarray:
    """Map global phase encoding to NS-local state: 0=green, 1=yellow, 2=red."""
    st = np.full(phases.shape, 2, dtype=int)
    st[phases == 0] = 0
    st[phases == 1] = 1
    return st


def _direction_state_ew(phases: np.ndarray) -> np.ndarray:
    """Map global phase encoding to EW-local state: 0=green, 1=yellow, 2=red."""
    st = np.full(phases.shape, 2, dtype=int)
    st[phases == 2] = 0
    st[phases == 3] = 1
    return st


def _pairwise_state_sync(state_matrix: np.ndarray) -> float:
    """
    Mean pairwise agreement ratio across intersections for one direction.
    1.0 means intersections are always in the same directional state.
    """
    n = int(state_matrix.shape[0])
    if n < 2:
        return float('nan')
    pair_scores = []
    for i in range(n):
        si = state_matrix[i]
        for j in range(i + 1, n):
            pair_scores.append(float(np.mean(si == state_matrix[j])))
    return float(np.mean(pair_scores)) if pair_scores else float('nan')


def _arrival_green_score(upstream_green: np.ndarray, downstream_green: np.ndarray, tau: int) -> float:
    """
    Fraction of upstream-green departures that reach downstream on green,
    using travel-time shift tau (seconds).
    """
    if tau < 0 or tau >= upstream_green.size:
        return float('nan')
    if tau == 0:
        up = upstream_green
        dn = downstream_green
    else:
        up = upstream_green[:-tau]
        dn = downstream_green[tau:]
    movers = up.astype(bool)
    if not np.any(movers):
        return float('nan')
    return float(np.mean(dn[movers]))


def _progression_sync_by_direction(green_matrix: np.ndarray, axis: str) -> float:
    """
    Green-wave progression score for adjacent intersections.

    axis='ew': pairs along each row (left/right neighbors), tau from SEGMENT_LENGTHS_V.
    axis='ns': pairs along each column (top/bottom neighbors), tau from SEGMENT_LENGTHS_H.
    """
    scores = []
    if axis == 'ew':
        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE - 1):
                a = row * GRID_SIZE + col
                b = row * GRID_SIZE + (col + 1)
                tau = max(1, int(round(SEGMENT_LENGTHS_V[col + 1] / SPEED_LIMIT)))
                scores.append(_arrival_green_score(green_matrix[a], green_matrix[b], tau))
                scores.append(_arrival_green_score(green_matrix[b], green_matrix[a], tau))
    elif axis == 'ns':
        for row in range(GRID_SIZE - 1):
            for col in range(GRID_SIZE):
                a = row * GRID_SIZE + col
                b = (row + 1) * GRID_SIZE + col
                tau = max(1, int(round(SEGMENT_LENGTHS_H[row + 1] / SPEED_LIMIT)))
                scores.append(_arrival_green_score(green_matrix[a], green_matrix[b], tau))
                scores.append(_arrival_green_score(green_matrix[b], green_matrix[a], tau))
    else:
        return float('nan')

    scores = [s for s in scores if not np.isnan(s)]
    return float(np.mean(scores)) if scores else float('nan')


def compute_signal_sync_metrics(schedules: dict) -> dict:
    """Compute directional synchronization metrics for one strategy schedule set."""
    phases = np.array([schedules[ix] for ix in range(N_INTERSECTIONS)], dtype=int)
    ns_state = _direction_state_ns(phases)
    ew_state = _direction_state_ew(phases)
    ns_green = (phases == 0)
    ew_green = (phases == 2)

    return {
        'ns_global_sync': _pairwise_state_sync(ns_state),
        'ew_global_sync': _pairwise_state_sync(ew_state),
        'ns_progression_sync': _progression_sync_by_direction(ns_green, axis='ns'),
        'ew_progression_sync': _progression_sync_by_direction(ew_green, axis='ew'),
    }


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
# Plot 1 — Phase-sequence heatmap (own window + file)
# ---------------------------------------------------------------------------
fig_phase, axes_phase = plt.subplots(
    GRID_SIZE,
    GRID_SIZE,
    figsize=(16, 10),
    layout='constrained',
    sharex=True,
    sharey=True,
)
draw_phase_heatmap(fig_phase, axes_phase, ev_schedules, bl_schedules, best_blocks)
_savefig_only(os.path.join(OUTPUT_DIR, 'plot_phase_sequence.png'))
plt.show()
print("Saved: plot_phase_sequence.png")
plt.close(fig_phase)

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
# Shared series for bar/line/scatter plots
# ---------------------------------------------------------------------------
vtypes = ['car', 'bus', 'truck', 'goal_car']
x = np.arange(len(vtypes))
width = 0.35
t_axis = np.arange(3600)


def avg_tt_by_type(finished, vt):
    vals = [v.travel_time for v in finished if v.vtype == vt]
    return np.mean(vals) if vals else 0


def avg_idl_by_type(vehicles, vt):
    vals = [v.idling_time for v in vehicles if v.vtype == vt]
    return np.mean(vals) if vals else 0


def finish_rate(vehicles, vt):
    total = [v for v in vehicles if v.vtype == vt]
    fin = [v for v in total if v.finished]
    return 100 * len(fin) / max(len(total), 1)


ev_tt = [avg_tt_by_type(ev_finished, vt) for vt in vtypes]
bl_tt = [avg_tt_by_type(bl_finished, vt) for vt in vtypes]
ev_idl = [avg_idl_by_type(ev_vehicles, vt) for vt in vtypes]
bl_idl = [avg_idl_by_type(bl_vehicles, vt) for vt in vtypes]
ev_fr = [finish_rate(ev_vehicles, vt) for vt in vtypes]
bl_fr = [finish_rate(bl_vehicles, vt) for vt in vtypes]

# ---------------------------------------------------------------------------
# Save plot 2 — vehicle breakdown (separate file, no interactive window)
# ---------------------------------------------------------------------------
_fig, _axes = plt.subplots(1, 2, figsize=(12, 5), layout='constrained')
draw_vehicle_breakdown(_axes[0], _axes[1], vtypes, x, width, ev_tt, bl_tt, ev_idl, bl_idl)
_fig.suptitle("Evolved vs Baseline: Travel & Idling Breakdown", fontsize=13)
_savefig_only(os.path.join(OUTPUT_DIR, 'plot_vehicle_breakdown.png'))
plt.close(_fig)
print("Saved: plot_vehicle_breakdown.png")

# ---------------------------------------------------------------------------
# Save plot 3 — finish rate
# ---------------------------------------------------------------------------
_fig, _ax = plt.subplots(figsize=(7, 5), layout='constrained')
draw_finish_rate(_ax, vtypes, x, width, ev_fr, bl_fr)
_fig.suptitle("Finish Rate by Vehicle Type (%)", fontsize=13)
_savefig_only(os.path.join(OUTPUT_DIR, 'plot_finish_rate.png'))
plt.close(_fig)
print("Saved: plot_finish_rate.png")

# ---------------------------------------------------------------------------
# Save plot 4 — density
# ---------------------------------------------------------------------------
_fig, _ax = plt.subplots(figsize=(12, 5), layout='constrained')
draw_density(_ax, t_axis, ev_density, bl_density)
_fig.suptitle("Gridlock Analysis: Active Vehicles Over Time", fontsize=14)
_savefig_only(os.path.join(OUTPUT_DIR, 'plot_density.png'))
plt.close(_fig)
print("Saved: plot_density.png")

print(f"\nFinal fitness - Evolved: {ev_score:.2f}  |  Baseline: {bl_score:.2f}  "
      f"|  Improvement: {bl_score - ev_score:+.2f}")

# ---------------------------------------------------------------------------
# Directional signal synchronization analysis
# ---------------------------------------------------------------------------
ev_sync = compute_signal_sync_metrics(ev_schedules)
bl_sync = compute_signal_sync_metrics(bl_schedules)

print("\nDirectional signal synchronization (0..1; higher = more synchronized):")
print("  Global phase alignment (same directional state across intersections)")
print(f"    NS  evolved={ev_sync['ns_global_sync']:.3f}  baseline={bl_sync['ns_global_sync']:.3f}  "
    f"delta={ev_sync['ns_global_sync'] - bl_sync['ns_global_sync']:+.3f}")
print(f"    EW  evolved={ev_sync['ew_global_sync']:.3f}  baseline={bl_sync['ew_global_sync']:.3f}  "
    f"delta={ev_sync['ew_global_sync'] - bl_sync['ew_global_sync']:+.3f}")
print("  Green-wave progression alignment (arrival-on-green for adjacent intersections)")
print(f"    NS  evolved={ev_sync['ns_progression_sync']:.3f}  baseline={bl_sync['ns_progression_sync']:.3f}  "
    f"delta={ev_sync['ns_progression_sync'] - bl_sync['ns_progression_sync']:+.3f}")
print(f"    EW  evolved={ev_sync['ew_progression_sync']:.3f}  baseline={bl_sync['ew_progression_sync']:.3f}  "
    f"delta={ev_sync['ew_progression_sync'] - bl_sync['ew_progression_sync']:+.3f}")

sync_labels = [
    'NS global',
    'EW global',
    'NS progression',
    'EW progression',
]
sync_ev_vals = [
    ev_sync['ns_global_sync'],
    ev_sync['ew_global_sync'],
    ev_sync['ns_progression_sync'],
    ev_sync['ew_progression_sync'],
]
sync_bl_vals = [
    bl_sync['ns_global_sync'],
    bl_sync['ew_global_sync'],
    bl_sync['ns_progression_sync'],
    bl_sync['ew_progression_sync'],
]

_fig, _ax = plt.subplots(figsize=(10, 5), layout='constrained')
draw_signal_sync_comparison(_ax, sync_labels, sync_ev_vals, sync_bl_vals)
_fig.suptitle("Directional signal synchronization: evolved vs baseline", fontsize=13)
_savefig_only(os.path.join(OUTPUT_DIR, 'plot_signal_sync.png'))
plt.close(_fig)
print("Saved: plot_signal_sync.png")

# ---------------------------------------------------------------------------
# Road-length vs timings (data + save plot 5)
# ---------------------------------------------------------------------------
def _mean_adjacent_segment_m(row: int, col: int) -> float:
    sv = SEGMENT_LENGTHS_V
    sh = SEGMENT_LENGTHS_H
    return float(sv[col] + sv[col + 1] + sh[row] + sh[row + 1]) / 4.0


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size < 2 or b.size < 2:
        return float("nan")
    if np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


seg_mean = []
y_ns, y_ew, y_off = [], [], []
for ix_id, timing in enumerate(best_blocks):
    r, c = divmod(ix_id, GRID_SIZE)
    seg_mean.append(_mean_adjacent_segment_m(r, c))
    y_ns.append(timing.g_ns)
    y_ew.append(timing.g_ew)
    y_off.append(getattr(timing, "offset", 0))

seg_mean = np.array(seg_mean, dtype=float)
y_ns = np.array(y_ns, dtype=float)
y_ew = np.array(y_ew, dtype=float)
y_off = np.array(y_off, dtype=float)

r_ns = _pearson(seg_mean, y_ns)
r_ew = _pearson(seg_mean, y_ew)
r_off = _pearson(seg_mean, y_off)
print(f"\nRoad-length vs timing (Pearson r):  g_ns={r_ns:+.3f}  g_ew={r_ew:+.3f}  offset={r_off:+.3f}")

scatter_kw = dict(s=70, alpha=0.88, edgecolors="black", linewidths=0.4)
# Slightly smaller markers in the tall dashboard so points stay legible when the window is resized.
scatter_kw_dash = {**scatter_kw, "s": 52}
_fig, _axes = plt.subplots(1, 3, figsize=(14, 4), layout='constrained')
draw_correlation_scatters(_axes, seg_mean, y_ns, y_ew, y_off, r_ns, r_ew, r_off, scatter_kw)
_fig.suptitle(
    "Evolved timings vs local road geometry (16 intersections; interpret weak r with caution)",
    fontsize=12,
)
_savefig_only(os.path.join(OUTPUT_DIR, "plot_road_timing_correlation.png"))
plt.close(_fig)
print("Saved: plot_road_timing_correlation.png")

# ---------------------------------------------------------------------------
# Single dashboard window: travel/idling | finish | density | correlation
# ---------------------------------------------------------------------------
_dash = plt.figure(figsize=(14, 20), layout='constrained')
try:
    _dash.canvas.manager.set_window_title("Traffic validation — metrics dashboard")
except Exception:
    pass

_gs = _dash.add_gridspec(5, 1, height_ratios=[1.05, 0.62, 0.72, 1.15, 1.0])
_gs0 = _gs[0].subgridspec(1, 2, wspace=0.06)
_ax_tt = _dash.add_subplot(_gs0[0, 0])
_ax_idl = _dash.add_subplot(_gs0[0, 1])
draw_vehicle_breakdown(_ax_tt, _ax_idl, vtypes, x, width, ev_tt, bl_tt, ev_idl, bl_idl)

_ax_fr = _dash.add_subplot(_gs[1])
draw_finish_rate(_ax_fr, vtypes, x, width, ev_fr, bl_fr)

_ax_sync = _dash.add_subplot(_gs[2])
draw_signal_sync_comparison(_ax_sync, sync_labels, sync_ev_vals, sync_bl_vals)

_ax_den = _dash.add_subplot(_gs[3])
draw_density(_ax_den, t_axis, ev_density, bl_density)

_gs3 = _gs[4].subgridspec(1, 3, wspace=0.1)
_ax_c0 = _dash.add_subplot(_gs3[0, 0])
_ax_c1 = _dash.add_subplot(_gs3[0, 1])
_ax_c2 = _dash.add_subplot(_gs3[0, 2])
draw_correlation_scatters(
    [_ax_c0, _ax_c1, _ax_c2],
    seg_mean,
    y_ns,
    y_ew,
    y_off,
    r_ns,
    r_ew,
    r_off,
    scatter_kw_dash,
)

_dash.suptitle(
    "Evolved vs baseline — validation dashboard (resize window; layout reflows automatically)",
    fontsize=14,
)
plt.show()
print("Validation dashboard closed.")
