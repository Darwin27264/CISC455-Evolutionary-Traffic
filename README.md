# Evolutionary optimization of traffic signal timing

**CISC 455.** This repository trains an evolutionary algorithm (EA) to set per-intersection traffic light parameters on a continuous **N×N** road grid with bidirectional traffic, car-following physics, and goal-directed vehicles routed with **A\***.

![Simulation demo](demo.gif)

## Overview

Urban signal timing is usually hand-tuned or fixed to a simple pattern. Here, each intersection's timing policy is **evolved**: the simulator scores how well vehicles move through the network, and the EA searches for better combinations of green times and phase offsets. The goal is to reduce wasted time at red lights while respecting mandatory yellow and all-red clearance intervals so the optimizer cannot invent unsafe transitions.

The active codebase is Python scripts (`ArrayBasedTraining.py`, `ArrayValidation.py`, `ArrayReplay.py`, `BestTimingSummary.py`, `routing.py`). Legacy Jupyter notebooks are kept under `LegacyExplorationNotebooks/` for reference only.

---

## Simulation model

**Road geometry.** The map spans `[0, MAP_END]` metres on each axis (default 1000 m). Horizontal roads lie at fixed **y** coordinates `INTERSECTIONS_H`; vertical roads at **x** coordinates `INTERSECTIONS_V`. The two lists differ on purpose so segment lengths are **not symmetric**: the network is harder to exploit with a single repeated timing pattern.

**Array-based traffic.** Each lane on each road segment is a 1D array of integer cell states (empty or occupied by a vehicle id). Vehicles advance in discrete one-second steps with acceleration limits and a safety buffer to the vehicle ahead. Red lights cap movement at stop lines. This representation makes dense traffic and queues explicit compared to a purely abstract "delay per link" model.

**Signal encoding.** A repeating cycle uses integer phases: north–south green, yellow, east–west green, yellow, with fixed **6 s** yellow and **3 s** all-red after each green so every legal chromosome includes the same safety margins.

**Demand.** Vehicles are injected using repeated random spawn attempts per simulated second (steady-state load). A configurable fraction of spawns are **goal cars**: they appear at interior positions and follow a shortest-path route on the intersection graph where edge cost is physical road length in metres (`routing.py`). Remaining spawns are edge-entering cars, buses, and trucks.

---

## Evolutionary algorithm

**Representation.** Each individual is a list of `IntersectionTiming(g_ns, g_ew, offset)` objects — one per intersection. `g_ns` and `g_ew` are the north–south and east–west green durations (seconds); `offset` shifts the phase of that intersection relative to others within its cycle. Yellow and all-red segments are fixed and not subject to evolution, ensuring every genotype produces a physically valid signal schedule.

**Fitness.** The EA **minimizes** a cost aggregated over all vehicles spawned during a one-hour simulation. Idling time is weighted double relative to raw travel time, so solutions that keep traffic flowing through greens are preferred over those that shift cost between phases without reducing queues. Vehicles that do not finish their trip within the horizon incur a penalty, providing a gradient in early generations when completion rates are low.

**Baseline.** Evolved individuals are always compared against a **uniform baseline**: every intersection uses 30 s north–south / 30 s east–west with zero offset. This isolates the benefit of differentiated timings and coordinated offsets from simply having a reasonable cycle length.

**Variation operators.**
- **Tournament selection** (`TOURNAMENT_K`): picks parents with enough pressure to improve but mild enough to preserve diversity.
- **One-point crossover**: swaps whole `IntersectionTiming` objects between two parents so yellow/all-red blocks are never split.
- **Mutation**: replaces one randomly chosen intersection with a freshly sampled random timing.
- **Elitism**: the best individual is copied unchanged into the next generation.
- **Parallel evaluation**: each generation dispatches fitness calls across CPU cores via `ProcessPoolExecutor`. All individuals and the baseline share the same stochastic traffic seed per generation for a fair comparison.

Default population size, generation count, and operator rates are defined at the top of `ArrayBasedTraining.py`.

---

## Running the project

Install dependencies (Pygame is only required for the visual replay):

```bash
pip install -r requirements.txt
```

Train a new run (writes `runs/<timestamp>/best_timing_array.pkl`, updates `runs/latest_run.txt`, and logs progress to that folder):

```bash
python ArrayBasedTraining.py
```

Validate a saved chromosome (replays evolved vs baseline under the same stochastic traffic seed, prints metrics, and saves figures to the run directory):

```bash
python ArrayValidation.py --pkl runs/<timestamp>/best_timing_array.pkl
```

Print a text summary of how evolved timings differ from the baseline:

```bash
python BestTimingSummary.py --pkl runs/<timestamp>/best_timing_array.pkl
```

`--top N` (default 8) controls how many most-changed intersections are listed. Omit `--pkl` to resolve the path automatically from `runs/latest_run.txt`.

Visual replay in Pygame (lighter traffic than training for clarity):

```bash
python ArrayReplay.py --pkl runs/<timestamp>/best_timing_array.pkl
```

---

## Outputs and how to read them

Figures are saved next to the pickle passed to validation (or under the latest run directory). The console prints normalized fitness for evolved and baseline replays and their difference.

| File | Meaning |
|------|---------|
| `plot_phase_sequence.png` | First 20 minutes of light phases per intersection: evolved vs baseline stacked. |
| `plot_vehicle_breakdown.png` | Mean travel time (finished vehicles) and mean idling (all spawned), broken down by vehicle type. |
| `plot_finish_rate.png` | Percentage of spawned vehicles that complete their trip within the one-hour horizon. |
| `plot_density.png` | Count of active vehicles each second (congestion proxy) with linear trend lines. |
| `plot_signal_sync.png` | Signal synchronization metrics: global pairwise phase alignment and green-wave progression along links, using travel-time shift as delay. |
| `plot_road_timing_correlation.png` | Exploratory scatter of local mean segment length vs evolved `g_ns`, `g_ew`, and offset, with Pearson **r**. |

**Interpreting results.** Any single run depends on seeds, spawn parameters, and EA settings; treat numbers as illustrative unless averaged over repeated experiments. For one run on `runs/2026-03-22_032423`, evolved timings achieved a normalized fitness of approximately **153 vs 180** for the baseline (lower is better), with high finish rates for goal-directed vehicles. Pearson **r** between local road length and evolved green times was weak (order of **0.2**), which is expected when only 16 intersections contribute one point each and timings are co-adapted across the whole network.

---

## Repository layout

| Path | Purpose |
|------|---------|
| `ArrayBasedTraining.py` | EA loop, array-based simulation, run directory creation, `best_timing_array.pkl` export |
| `ArrayValidation.py` | Load pickle, deterministic replay vs baseline, metrics and figures |
| `ArrayReplay.py` | Pygame visualization of a saved solution |
| `BestTimingSummary.py` | CLI text summary: evolved vs baseline timing differences and sync metrics |
| `routing.py` | A\*, route legs, interior spawn sampling |
| `runs/` | Timestamped training outputs and `latest_run.txt` |
| `LegacyExplorationNotebooks/` | Archived exploratory notebooks (earlier pipeline, not required) |

---

## Requirements

- **Python** 3.10+ recommended.
- **numpy**, **matplotlib** — training and validation.
- **pygame** — `ArrayReplay.py` only; training and validation run headless without it.

Jupyter is optional and only needed if you open the legacy notebooks.
