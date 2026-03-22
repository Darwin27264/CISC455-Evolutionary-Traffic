"""
routing.py
----------
A* path planning on the road-intersection grid for goal-directed vehicles.

The simulation uses GRID_SIZE horizontal roads (y = INTERSECTIONS_H[i]) and
GRID_SIZE vertical roads (x = INTERSECTIONS_V[j]). Intersection (i, j) is at
(x, y) = (INTERSECTIONS_V[j], INTERSECTIONS_H[i]).

Virtual start (S) and goal (G) nodes connect to all intersections on the
spawn/goal road with cost equal to driving distance along that road.
"""

from __future__ import annotations

import heapq
import math
import random
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

# Virtual node ids (must not collide with flattened grid indices)
NODE_S = -1
NODE_G = -2


@dataclass(frozen=True)
class RoadPosition:
    """A point on the network: one axis, channel index, coordinate along that axis."""

    axis: str  # 'h' or 'v'
    channel_idx: int
    abs_pos: float


@dataclass
class DriveLeg:
    """One straight segment along a single road until end_abs (inclusive target)."""

    axis: str
    channel_idx: int
    start_abs: float
    end_abs: float
    is_final: bool


def _flatten(hr: int, vc: int, grid_size: int) -> int:
    return hr * grid_size + vc


def _unflatten(idx: int, grid_size: int) -> Tuple[int, int]:
    return idx // grid_size, idx % grid_size


def _intersection_xy(
    hr: int, vc: int, intersections_h: Sequence[float], intersections_v: Sequence[float]
) -> Tuple[float, float]:
    return float(intersections_v[vc]), float(intersections_h[hr])


def sample_interior_position(
    rng: random.Random,
    grid_size: int,
    intersections_h: np.ndarray,
    intersections_v: np.ndarray,
    map_end: float,
    margin: float = 8.0,
) -> RoadPosition:
    """
    Pick a random point on a random road segment that is not on the map boundary
    (i.e. not in the first or last segment along that axis).
    """
    axis = rng.choice(["h", "v"])
    ix = intersections_v if axis == "h" else intersections_h
    boundaries = [0.0] + [float(x) for x in ix] + [float(map_end)]
    n_seg = len(boundaries) - 1
    inner = [i for i in range(n_seg) if i not in (0, n_seg - 1)]
    if not inner:
        seg_i = max(0, n_seg // 2)
    else:
        seg_i = rng.choice(inner)
    lo, hi = boundaries[seg_i], boundaries[seg_i + 1]
    span = hi - lo
    safe = max(margin * 2.0, span * 0.1)
    if span <= 2.0 * safe:
        abs_pos = (lo + hi) / 2.0
    else:
        abs_pos = rng.uniform(lo + safe, hi - safe)
    abs_pos = max(0.0, min(float(map_end), abs_pos))
    ch = rng.randrange(grid_size)
    return RoadPosition(axis, ch, abs_pos)


def _astar(
    grid_size: int,
    intersections_h: Sequence[float],
    intersections_v: Sequence[float],
    spawn: RoadPosition,
    goal: RoadPosition,
) -> Optional[List[int]]:
    """
    Returns list of flattened intersection indices from first hop after S to last before G,
    or None if unreachable.
    """
    n_nodes = grid_size * grid_size

    def neigh_intersection(idx: int) -> List[Tuple[int, float]]:
        hr, vc = _unflatten(idx, grid_size)
        out: List[Tuple[int, float]] = []
        if vc + 1 < grid_size:
            w = abs(intersections_v[vc + 1] - intersections_v[vc])
            out.append((_flatten(hr, vc + 1, grid_size), w))
        if vc > 0:
            w = abs(intersections_v[vc] - intersections_v[vc - 1])
            out.append((_flatten(hr, vc - 1, grid_size), w))
        if hr + 1 < grid_size:
            w = abs(intersections_h[hr + 1] - intersections_h[hr])
            out.append((_flatten(hr + 1, vc, grid_size), w))
        if hr > 0:
            w = abs(intersections_h[hr] - intersections_h[hr - 1])
            out.append((_flatten(hr - 1, vc, grid_size), w))
        return out

    def dist_to_goal_node(idx: int) -> float:
        hr, vc = _unflatten(idx, grid_size)
        x, y = _intersection_xy(hr, vc, intersections_h, intersections_v)
        if goal.axis == "h":
            gx = goal.abs_pos
            gy = float(intersections_h[goal.channel_idx])
            return abs(gx - x) + abs(gy - y)
        gx = float(intersections_v[goal.channel_idx])
        gy = goal.abs_pos
        return abs(gx - x) + abs(gy - y)

    # Edges from S
    def edges_from_s() -> List[Tuple[int, float]]:
        out: List[Tuple[int, float]] = []
        if spawn.axis == "h":
            hr = spawn.channel_idx
            xs = spawn.abs_pos
            for vc in range(grid_size):
                ix = float(intersections_v[vc])
                out.append((_flatten(hr, vc, grid_size), abs(ix - xs)))
        else:
            vc = spawn.channel_idx
            ys = spawn.abs_pos
            for hr in range(grid_size):
                iy = float(intersections_h[hr])
                out.append((_flatten(hr, vc, grid_size), abs(iy - ys)))
        return out

    # Edges to G (from intersection idx)
    def edge_to_g(idx: int) -> float:
        hr, vc = _unflatten(idx, grid_size)
        x, y = _intersection_xy(hr, vc, intersections_h, intersections_v)
        if goal.axis == "h":
            return abs(goal.abs_pos - x) + abs(float(intersections_h[goal.channel_idx]) - y)
        return abs(float(intersections_v[goal.channel_idx]) - x) + abs(goal.abs_pos - y)

    # A* : node ids are NODE_S, NODE_G, or 0..n_nodes-1
    open_heap: List[Tuple[float, float, int, int]] = []
    g_score: dict = {NODE_S: 0.0}
    parent: dict = {}

    def h(u: int) -> float:
        if u == NODE_G:
            return 0.0
        if u == NODE_S:
            x_s, y_s = (
                (spawn.abs_pos, float(intersections_h[spawn.channel_idx]))
                if spawn.axis == "h"
                else (float(intersections_v[spawn.channel_idx]), spawn.abs_pos)
            )
            if goal.axis == "h":
                gx, gy = goal.abs_pos, float(intersections_h[goal.channel_idx])
            else:
                gx, gy = float(intersections_v[goal.channel_idx]), goal.abs_pos
            return abs(gx - x_s) + abs(gy - y_s)

        return dist_to_goal_node(u)

    heapq.heappush(open_heap, (h(NODE_S), 0.0, NODE_S, NODE_S))

    while open_heap:
        _, g_u, u, _ = heapq.heappop(open_heap)
        if g_u > g_score.get(u, math.inf):
            continue
        if u == NODE_G:
            # Reconstruct: G -> ... -> S
            path_rev: List[int] = []
            cur = u
            while cur != NODE_S:
                p = parent.get(cur)
                if p is None:
                    return None
                if p != NODE_S:
                    path_rev.append(p)
                cur = p
            path_rev.reverse()
            return path_rev

        if u == NODE_S:
            for v, w in edges_from_s():
                tg = g_u + w
                if tg < g_score.get(v, math.inf):
                    g_score[v] = tg
                    parent[v] = u
                    f = tg + h(v)
                    heapq.heappush(open_heap, (f, tg, v, u))
            continue

        if 0 <= u < n_nodes:
            for v, w in neigh_intersection(u):
                tg = g_u + w
                if tg < g_score.get(v, math.inf):
                    g_score[v] = tg
                    parent[v] = u
                    f = tg + h(v)
                    heapq.heappush(open_heap, (f, tg, v, u))
            tg = g_u + edge_to_g(u)
            if tg < g_score.get(NODE_G, math.inf):
                g_score[NODE_G] = tg
                parent[NODE_G] = u
                f = tg + h(NODE_G)
                heapq.heappush(open_heap, (f, tg, NODE_G, u))

    return None


def plan_drive_legs(
    grid_size: int,
    intersections_h: Sequence[float],
    intersections_v: Sequence[float],
    map_end: float,
    spawn: RoadPosition,
    goal: RoadPosition,
) -> Optional[Tuple[List[DriveLeg], List[Tuple[float, float]]]]:
    """
    Plan straight legs from spawn to goal. Returns (legs, path_polyline_xy) or None.
    Polyline is in simulation coordinates (x along width, y along height) for drawing.
    """
    node_path = _astar(grid_size, intersections_h, intersections_v, spawn, goal)
    if node_path is None:
        return None

    legs: List[DriveLeg] = []
    poly: List[Tuple[float, float]] = []

    def append_point(ax: str, c: int, a: float) -> None:
        if ax == "h":
            poly.append((a, float(intersections_h[c])))
        else:
            poly.append((float(intersections_v[c]), a))

    axis = spawn.axis
    ch = spawn.channel_idx
    pos = spawn.abs_pos
    append_point(axis, ch, pos)

    def go_straight_to(target_abs: float, final: bool) -> None:
        nonlocal pos, axis, ch
        start = pos
        legs.append(DriveLeg(axis, ch, start, target_abs, final))
        pos = target_abs
        append_point(axis, ch, pos)

    ih = [float(x) for x in intersections_h]
    iv = [float(x) for x in intersections_v]

    i = 0
    n = len(node_path)

    while i < n:
        hr, vc = _unflatten(node_path[i], grid_size)
        ix, iy = iv[vc], ih[hr]

        if axis == "h":
            if ch != hr:
                return None
            go_straight_to(ix, final=False)
            nxt_hr, nxt_vc = hr, vc
            if i + 1 < n:
                nxt_hr, nxt_vc = _unflatten(node_path[i + 1], grid_size)
            if nxt_hr == hr:
                i += 1
                continue
            axis, ch, pos = "v", vc, iy
            append_point(axis, ch, pos)
            i += 1
        else:
            if ch != vc:
                return None
            go_straight_to(iy, final=False)
            nxt_hr, nxt_vc = hr, vc
            if i + 1 < n:
                nxt_hr, nxt_vc = _unflatten(node_path[i + 1], grid_size)
            if nxt_vc == vc:
                i += 1
                continue
            axis, ch, pos = "h", hr, ix
            append_point(axis, ch, pos)
            i += 1

    # Final approach to goal.
    # May require driving straight on the current road then one turn, depending
    # on whether we're already on the goal's axis+channel.
    if goal.axis == "h":
        if axis == "h" and ch == goal.channel_idx:
            go_straight_to(goal.abs_pos, final=True)
        elif axis == "v":
            # On v-road ch (index into V); need to reach h-road goal.channel_idx.
            # Drive along this v-road to y = ih[goal.channel_idx], then turn onto h-road.
            gy = ih[goal.channel_idx]
            gx_here = iv[ch]  # x-position of the current v-road
            go_straight_to(gy, final=False)
            axis, ch, pos = "h", goal.channel_idx, gx_here
            append_point(axis, ch, pos)
            go_straight_to(goal.abs_pos, final=True)
        else:
            # On a different h-road — shouldn't happen if A* is correct, but guard.
            return None
    else:
        if axis == "v" and ch == goal.channel_idx:
            go_straight_to(goal.abs_pos, final=True)
        elif axis == "h":
            # On h-road ch (index into H); need to reach v-road goal.channel_idx.
            gx = iv[goal.channel_idx]
            gy_here = ih[ch]  # y-position of the current h-road
            go_straight_to(gx, final=False)
            axis, ch, pos = "v", goal.channel_idx, gy_here
            append_point(axis, ch, pos)
            go_straight_to(goal.abs_pos, final=True)
        else:
            return None

    # Filter out zero-length legs (start ≈ end) that can arise when the vehicle
    # is already at the target intersection.  Keep only legs with meaningful travel.
    filtered = [lg for lg in legs if abs(lg.end_abs - lg.start_abs) > 0.5]
    if not filtered:
        return None
    filtered[-1].is_final = True
    return filtered, poly


def try_plan_goal_route(
    rng: random.Random,
    grid_size: int,
    intersections_h: np.ndarray,
    intersections_v: np.ndarray,
    map_end: float,
    max_sample_tries: int = 40,
) -> Optional[Tuple[RoadPosition, RoadPosition, List[DriveLeg], List[Tuple[float, float]]]]:
    """Sample spawn/goal and return a feasible plan, or None."""
    for _ in range(max_sample_tries):
        spawn = sample_interior_position(rng, grid_size, intersections_h, intersections_v, map_end)
        goal = sample_interior_position(rng, grid_size, intersections_h, intersections_v, map_end)
        if (
            spawn.axis == goal.axis
            and spawn.channel_idx == goal.channel_idx
            and abs(spawn.abs_pos - goal.abs_pos) < 15.0
        ):
            continue
        planned = plan_drive_legs(
            grid_size, intersections_h, intersections_v, map_end, spawn, goal
        )
        if planned is not None:
            legs, poly = planned
            return spawn, goal, legs, poly
    return None
