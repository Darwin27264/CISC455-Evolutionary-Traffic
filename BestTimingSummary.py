"""
BestTimingSummary.py
--------------------
Prints a concise baseline-vs-best timing summary for a trained chromosome.

Default behavior loads runs/latest_run.txt -> runs/<timestamp>/best_timing_array.pkl.
You can override with --pkl <path>.
"""

from __future__ import annotations

import argparse
import os
import pickle
import __main__
from pathlib import Path

import numpy as np

from ArrayBasedTraining import (
    TimingBlock,
    IntersectionTiming,
    GRID_SIZE,
    N_INTERSECTIONS,
    SEGMENT_LENGTHS_H,
    SEGMENT_LENGTHS_V,
    SPEED_LIMIT,
    build_baseline_genes,
    build_schedules,
    resolve_pkl_path,
)

# Some training pickles were serialized with classes under __main__.
__main__.TimingBlock = TimingBlock
__main__.IntersectionTiming = IntersectionTiming


def _direction_state_ns(phases: np.ndarray) -> np.ndarray:
    st = np.full(phases.shape, 2, dtype=int)
    st[phases == 0] = 0
    st[phases == 1] = 1
    return st


def _direction_state_ew(phases: np.ndarray) -> np.ndarray:
    st = np.full(phases.shape, 2, dtype=int)
    st[phases == 2] = 0
    st[phases == 3] = 1
    return st


def _pairwise_state_sync(state_matrix: np.ndarray) -> float:
    n = int(state_matrix.shape[0])
    if n < 2:
        return float("nan")
    pair_scores = []
    for i in range(n):
        for j in range(i + 1, n):
            pair_scores.append(float(np.mean(state_matrix[i] == state_matrix[j])))
    return float(np.mean(pair_scores)) if pair_scores else float("nan")


def _arrival_green_score(upstream_green: np.ndarray, downstream_green: np.ndarray, tau: int) -> float:
    if tau < 0 or tau >= upstream_green.size:
        return float("nan")
    up = upstream_green if tau == 0 else upstream_green[:-tau]
    dn = downstream_green if tau == 0 else downstream_green[tau:]
    movers = up.astype(bool)
    if not np.any(movers):
        return float("nan")
    return float(np.mean(dn[movers]))


def _progression_sync_by_direction(green_matrix: np.ndarray, axis: str) -> float:
    scores = []
    if axis == "ew":
        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE - 1):
                a = row * GRID_SIZE + col
                b = row * GRID_SIZE + (col + 1)
                tau = max(1, int(round(SEGMENT_LENGTHS_V[col + 1] / SPEED_LIMIT)))
                scores.append(_arrival_green_score(green_matrix[a], green_matrix[b], tau))
                scores.append(_arrival_green_score(green_matrix[b], green_matrix[a], tau))
    elif axis == "ns":
        for row in range(GRID_SIZE - 1):
            for col in range(GRID_SIZE):
                a = row * GRID_SIZE + col
                b = (row + 1) * GRID_SIZE + col
                tau = max(1, int(round(SEGMENT_LENGTHS_H[row + 1] / SPEED_LIMIT)))
                scores.append(_arrival_green_score(green_matrix[a], green_matrix[b], tau))
                scores.append(_arrival_green_score(green_matrix[b], green_matrix[a], tau))
    else:
        return float("nan")

    scores = [s for s in scores if not np.isnan(s)]
    return float(np.mean(scores)) if scores else float("nan")


def _compute_signal_sync_metrics(schedules: dict[int, np.ndarray]) -> dict[str, float]:
    phases = np.array([schedules[ix] for ix in range(N_INTERSECTIONS)], dtype=int)
    ns_state = _direction_state_ns(phases)
    ew_state = _direction_state_ew(phases)
    ns_green = phases == 0
    ew_green = phases == 2
    return {
        "ns_global_sync": _pairwise_state_sync(ns_state),
        "ew_global_sync": _pairwise_state_sync(ew_state),
        "ns_progression_sync": _progression_sync_by_direction(ns_green, axis="ns"),
        "ew_progression_sync": _progression_sync_by_direction(ew_green, axis="ew"),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Print baseline-vs-best timing summary.")
    parser.add_argument(
        "--pkl",
        type=str,
        default=None,
        help="Path to best_timing_array.pkl (default resolves from runs/latest_run.txt)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=8,
        help="How many most-changed intersections to print (default: 8)",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    pkl_path = os.path.abspath(resolve_pkl_path(args.pkl))
    with open(pkl_path, "rb") as f:
        best_blocks = pickle.load(f)

    baseline = build_baseline_genes()
    base_g_ns = float(np.mean([b.g_ns for b in baseline]))
    base_g_ew = float(np.mean([b.g_ew for b in baseline]))
    base_off = float(np.mean([getattr(b, "offset", 0) for b in baseline]))

    g_ns = np.array([t.g_ns for t in best_blocks], dtype=float)
    g_ew = np.array([t.g_ew for t in best_blocks], dtype=float)
    off = np.array([getattr(t, "offset", 0) for t in best_blocks], dtype=float)

    ev_sync = _compute_signal_sync_metrics(build_schedules(best_blocks))
    bl_sync = _compute_signal_sync_metrics(build_schedules(baseline))

    print(Path(pkl_path))
    print()
    print("1. Green durations got shorter overall")
    print(
        f"- NS green: mean {g_ns.mean():.2f}s vs baseline {base_g_ns:.0f}s, "
        f"so about {g_ns.mean() - base_g_ns:+.2f}s on average"
    )
    print(
        f"- EW green: mean {g_ew.mean():.2f}s vs baseline {base_g_ew:.0f}s, "
        f"so about {g_ew.mean() - base_g_ew:+.2f}s on average"
    )
    print("- Range across intersections:")
    print(f"- NS: {g_ns.min():.0f}s to {g_ns.max():.0f}s")
    print(f"- EW: {g_ew.min():.0f}s to {g_ew.max():.0f}s")

    print("2. Offsets are now heavily used")
    print(f"- Mean offset: {off.mean():.2f}s")
    print(f"- Range: {off.min():.0f}s to {off.max():.0f}s")
    print(f"- {int(np.count_nonzero(off))} out of {len(off)} intersections have nonzero offset")
    print(f"- Baseline had all offsets at {base_off:.0f}s")

    print("3. Sync behavior vs baseline")
    print(
        f"- NS global sync: {ev_sync['ns_global_sync']:.3f} vs {bl_sync['ns_global_sync']:.3f} baseline "
        f"(delta {ev_sync['ns_global_sync'] - bl_sync['ns_global_sync']:+.3f})"
    )
    print(
        f"- EW global sync: {ev_sync['ew_global_sync']:.3f} vs {bl_sync['ew_global_sync']:.3f} baseline "
        f"(delta {ev_sync['ew_global_sync'] - bl_sync['ew_global_sync']:+.3f})"
    )
    print(
        f"- NS progression sync: {ev_sync['ns_progression_sync']:.3f} vs {bl_sync['ns_progression_sync']:.3f} baseline "
        f"(delta {ev_sync['ns_progression_sync'] - bl_sync['ns_progression_sync']:+.3f})"
    )
    print(
        f"- EW progression sync: {ev_sync['ew_progression_sync']:.3f} vs {bl_sync['ew_progression_sync']:.3f} baseline "
        f"(delta {ev_sync['ew_progression_sync'] - bl_sync['ew_progression_sync']:+.3f})"
    )

    print("4. Most changed intersections (largest total deviation)")
    rows = []
    for ix_id, timing in enumerate(best_blocks):
        row, col = divmod(ix_id, GRID_SIZE)
        dns = int(round(timing.g_ns - base_g_ns))
        dew = int(round(timing.g_ew - base_g_ew))
        dof = int(round(getattr(timing, "offset", 0) - base_off))
        score = abs(dns) + abs(dew) + abs(dof)
        rows.append((score, row, col, int(timing.g_ns), int(timing.g_ew), dof, dns, dew))

    rows.sort(reverse=True)
    for _, row, col, gns, gew, dof, dns, dew in rows[: max(1, args.top)]:
        print(
            f"- ({row},{col}): NS {gns} ({dns:+d}), EW {gew} ({dew:+d}), off {dof:+d}"
        )


if __name__ == "__main__":
    main()
