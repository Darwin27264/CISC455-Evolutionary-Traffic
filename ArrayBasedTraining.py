"""
ArrayBasedTraining.py
---------------------
Alternate implementation of the traffic simulation using explicit physical road arrays.

Road representation:
  - The grid has GRID_SIZE x GRID_SIZE intersections.
  - Between consecutive intersections (and from the map edge to the first / last
    intersection) there is a road SEGMENT.
  - Every segment is stored as TWO 1-D numpy arrays of integers, one per direction:
        road_arrays[(axis, seg_idx, direction)]  →  np.ndarray of length seg_len (meters)
    where:
        axis      : 'h' (horizontal) or 'v' (vertical)
        seg_idx   : index of the segment (0 … GRID_SIZE, inclusive of edge segments)
        direction : 1  (forward,  left→right / top→bottom)
                    -1 (backward, right→left / bottom→top)
  - Each cell in the array holds:
        0          → empty (no vehicle present)
        vehicle-id → every meter occupied by that vehicle's body is filled
                      with its id (front cell through to the rear cell)

Traffic-light encoding (same as original):
    0: NS green,  EW red
    1: NS yellow, EW red
    2: NS red,    EW green
    3: NS red,    EW yellow
    4: both red
"""

import bisect
import numpy as np
import random
import pickle
import copy
import os
import sys
from datetime import datetime
from typing import List, Optional, Tuple

from routing import DriveLeg, RoadPosition, try_plan_goal_route

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
GRID_SIZE         = 4     # NxN intersections
HEADLESS_MODE     = True  # No pygame dependency for this script
SPAWN_RATE        = 0.25  # Probability per attempt of a vehicle spawning
SPAWNS_PER_SECOND = 4     # How many spawn attempts per simulation second
SPEED_LIMIT       = 10    # metres per second

# Optional back-pressure: stop injecting vehicles when the grid is already saturated.
# None = never throttle (maximum stress for the EA; queues can grow without bound).
# Set to e.g. 280 for demos / sanity checks so active count reaches a steadier plateau.
MAX_ACTIVE_VEHICLES_FOR_SPAWN: Optional[int] = None

# Goal-directed vehicles: fraction of spawn attempts routed to interior spawn + A* route.
# Each attempt samples uniformly in [MIN, MAX] for extra training variability (set equal for fixed rate).
GOAL_VEHICLE_FRACTION_MIN = 0.40
GOAL_VEHICLE_FRACTION_MAX = 0.60

# EA parameters
POP_SIZE      = 25
GENS          = 15
MUTATION_RATE = 0.3
CROSSOVER_RATE= 0.8
TOURNAMENT_K  = 3

# Intersection positions along each axis — H and V are intentionally different
# so the grid is asymmetric (more realistic, harder for the EA to exploit symmetry).
# H = y-coordinates of horizontal roads  (vehicles on h roads stop at V intersections)
# V = x-coordinates of vertical roads    (vehicles on v roads stop at H intersections)
np.random.seed(42)
INTERSECTIONS_H = sorted((np.array([150, 280, 450, 650]) + np.random.randint(-20, 20, GRID_SIZE)).tolist())
INTERSECTIONS_V = sorted((np.array([180, 350, 500, 720]) + np.random.randint(-20, 20, GRID_SIZE)).tolist())
# Keep a combined alias used only for road-array building (channel counts)
INTERSECTIONS = INTERSECTIONS_H  # legacy alias — do not use for stop-line logic

# ---------------------------------------------------------------------------
# Build road-segment lengths from intersection positions
# ---------------------------------------------------------------------------
# Each axis now has its own segment layout.
#   H axis: seg boundaries come from INTERSECTIONS_H (y-coords of h-roads)
#            h-road vehicles travel along x, crossing vertical roads at INTERSECTIONS_V
#   V axis: seg boundaries come from INTERSECTIONS_V (x-coords of v-roads)
#            v-road vehicles travel along y, crossing horizontal roads at INTERSECTIONS_H
MAP_END = 1000

# Forward scan for car-following (metres). Must exceed ~1 s free-run at vmax (10 m/s)
# or leaders stay “invisible” until dangerously close; 120 m is a safe default.
CAR_FOLLOW_LOOKAHEAD_M = min(MAP_END, 120)

def _build_segs(intersections):
    """Return (lengths, starts) for a given intersection list."""
    boundaries = [0] + intersections + [MAP_END]
    lengths = [max(1, int(boundaries[i+1] - boundaries[i])) for i in range(len(boundaries)-1)]
    starts  = [0] + [int(x) for x in intersections]
    return lengths, starts

SEGMENT_LENGTHS_H, SEGMENT_STARTS_H = _build_segs(INTERSECTIONS_H)
SEGMENT_LENGTHS_V, SEGMENT_STARTS_V = _build_segs(INTERSECTIONS_V)

# Legacy aliases (used by build_road_arrays which iterates both axes)
SEGMENT_LENGTHS = SEGMENT_LENGTHS_H
SEGMENT_STARTS  = SEGMENT_STARTS_H

# Precomputed segment boundaries for O(log n) pos_to_seg via bisect.
# H-axis vehicles travel along x and are divided by INTERSECTIONS_V crossings.
# V-axis vehicles travel along y and are divided by INTERSECTIONS_H crossings.
_BOUNDARIES_FOR_H = [0] + list(INTERSECTIONS_V) + [MAP_END]
_BOUNDARIES_FOR_V = [0] + list(INTERSECTIONS_H) + [MAP_END]

# ---------------------------------------------------------------------------
# Road array factory
# ---------------------------------------------------------------------------

def build_road_arrays():
    """
    Return a dict keyed by (axis, channel_idx, seg_idx, direction) → np.zeros(seg_len).
    H roads travel along x, so their segments are divided by INTERSECTIONS_V crossings.
    V roads travel along y, so their segments are divided by INTERSECTIONS_H crossings.
    """
    arrays = {}
    for axis in ('h', 'v'):
        seg_lengths = SEGMENT_LENGTHS_V if axis == 'h' else SEGMENT_LENGTHS_H
        n_channels  = GRID_SIZE  # one road per intersection on the perpendicular axis
        for channel_idx in range(n_channels):
            for seg_idx, seg_len in enumerate(seg_lengths):
                for direction in (1, -1):
                    arrays[(axis, channel_idx, seg_idx, direction)] = np.zeros(seg_len, dtype=int)
    return arrays

# ---------------------------------------------------------------------------
# Helper: convert absolute position → (seg_idx, local_index)
# ---------------------------------------------------------------------------

def pos_to_seg(abs_pos, axis='h'):
    """
    Given an absolute position along an axis (0…MAP_END) return
    (seg_idx, local_idx) where local_idx is the offset inside that segment.
    axis='h' → segments divided by INTERSECTIONS_V (x-crossings for h-roads)
    axis='v' → segments divided by INTERSECTIONS_H (y-crossings for v-roads)
    """
    boundaries  = _BOUNDARIES_FOR_H if axis == 'h' else _BOUNDARIES_FOR_V
    seg_lengths = SEGMENT_LENGTHS_V if axis == 'h' else SEGMENT_LENGTHS_H
    i = bisect.bisect_right(boundaries, abs_pos) - 1
    i = max(0, min(i, len(seg_lengths) - 1))
    local = int(abs_pos - boundaries[i])
    return i, max(0, min(local, seg_lengths[i] - 1))

def seg_to_pos(seg_idx, local_idx, axis='h'):
    """Inverse of pos_to_seg."""
    starts = SEGMENT_STARTS_V if axis == 'h' else SEGMENT_STARTS_H
    return starts[seg_idx] + local_idx

# ---------------------------------------------------------------------------
# Car-following (shared by Vehicle and GoalVehicle)
# ---------------------------------------------------------------------------

SAFETY_BUFFER_M = 2.0  # metres; keep below dist_to_front to avoid overlap


def _speed_from_headway(
    dist_to_front: float,
    speed: float,
    max_speed: float,
    accel: float,
    safety_buffer: float = SAFETY_BUFFER_M,
) -> Tuple[float, bool]:
    """
    One-second update: speed capped by how much space exists beyond the buffer.

    The old rule ``speed = max(0, dist - buffer)`` whenever ``dist <= speed + buffer``
    froze vehicles whenever ``0 < dist <= buffer`` (e.g. 1 m short of a stop line
    or leg end), so platoons never woke up after the light turned green.

    Returns (new_speed, True) if we must hold (no room past buffer), else (new_speed, False).
    """
    cap = dist_to_front - safety_buffer
    if cap <= 0:
        return 0.0, True
    new_speed = min(float(max_speed), speed + float(accel), cap)
    return new_speed, False


def _speed_goal_vehicle(
    dist_traffic: float,
    d_leg: float,
    speed: float,
    max_speed: float,
    accel: float,
    safety_buffer: float = SAFETY_BUFFER_M,
) -> Tuple[float, bool]:
    """
    Goal vehicles: lights and car-following use the bumper buffer; the leg endpoint
    (intersection / turn) does not — subtracting the buffer from d_leg made
    cap = d_leg - 2 <= 0 whenever the car was within 2 m of its turn, so they
    never moved again even with a green light.
    """
    if dist_traffic == float("inf"):
        cap_traffic = float("inf")
    else:
        cap_traffic = dist_traffic - safety_buffer

    cap_leg = max(0.0, d_leg)

    if cap_traffic == float("inf"):
        cap = cap_leg
    else:
        cap = min(max(0.0, cap_traffic), cap_leg)

    if cap <= 0:
        return 0.0, True
    new_speed = min(float(max_speed), speed + float(accel), cap)
    return new_speed, False


# ---------------------------------------------------------------------------
# Vehicle
# ---------------------------------------------------------------------------

# Vehicle type parameters
VTYPES = {
    'car':   {'length': 1, 'accel': 2, 'max_speed': SPEED_LIMIT},
    'bus':   {'length': 3, 'accel': 1, 'max_speed': SPEED_LIMIT},
    'truck': {'length': 5, 'accel': 1, 'max_speed': SPEED_LIMIT},
    # Same dynamics as car; uses A* legs + turns (see GoalVehicle)
    'goal_car': {'length': 1, 'accel': 2, 'max_speed': SPEED_LIMIT},
}

class Vehicle:
    """
    A vehicle that moves along physical road arrays.

    Attributes
    ----------
    abs_pos : float
        Absolute position along the axis in metres.
    seg_idx : int
        Which road segment the front of the vehicle is currently in.
    local_idx : int
        Cell index within the current segment.
    channel_idx : int
        Which parallel road the vehicle is on (index into INTERSECTIONS).
    """

    _next_id = 1  # class-level counter; reset externally each simulation run

    def __init__(
        self,
        vtype: str,
        axis: str,
        direction: int,
        channel_idx: int,
        abs_pos_override: Optional[float] = None,
    ):
        self.id        = Vehicle._next_id
        Vehicle._next_id += 1

        params         = VTYPES[vtype]
        self.vtype     = vtype
        self.length    = params['length']
        self.accel     = params['accel']
        self.max_speed = params['max_speed']

        self.axis        = axis        # 'h' or 'v'
        self.direction   = direction   # 1 or -1
        self.channel_idx = channel_idx # index into INTERSECTIONS

        # Spawn at the boundary and at top speed, unless overridden (interior spawn)
        if abs_pos_override is not None:
            self.abs_pos = float(abs_pos_override)
        else:
            self.abs_pos = 0.0 if direction == 1 else float(MAP_END)
        self.speed   = float(self.max_speed)

        self.travel_time = 0
        self.idling_time = 0
        self.finished    = False

    # ------------------------------------------------------------------
    # Array footprint helpers
    # ------------------------------------------------------------------

    def _cells(self) -> list:
        """Return list of (axis, channel_idx, direction, seg_idx, local_idx) for every body cell."""
        cells = []
        pos = self.abs_pos
        for _ in range(self.length):
            si, li = pos_to_seg(pos, self.axis)
            li_c = max(0, li)
            cells.append((self.axis, self.channel_idx, self.direction, si, li_c))
            pos -= self.direction
        return cells

    def stamp(self, road_arrays):
        """Stamp vehicle id into road arrays."""
        for ax, ch, d, si, li in self._cells():
            key = (ax, ch, si, d)
            if key in road_arrays:
                arr = road_arrays[key]
                idx = min(li, len(arr) - 1)
                arr[idx] = self.id

    def erase(self, road_arrays):
        """Clear only cells that still contain this vehicle's id (safe against overwrites)."""
        for ax, ch, d, si, li in self._cells():
            key = (ax, ch, si, d)
            if key in road_arrays:
                arr = road_arrays[key]
                idx = min(li, len(arr) - 1)
                if arr[idx] == self.id:
                    arr[idx] = 0

    # ------------------------------------------------------------------
    # Movement
    # ------------------------------------------------------------------

    def step(self, road_arrays, dist_to_front: float):
        """Advance vehicle by one second."""
        if self.finished:
            return

        # dist_to_front = min(gap to vehicle ahead, distance to stop line if red)
        self.speed, hold = _speed_from_headway(
            dist_to_front, self.speed, self.max_speed, self.accel
        )
        if hold:
            self.idling_time += 1

        if self.speed == 0:
            self.travel_time += 1
            return

        # --- Erase old position ---
        self.erase(road_arrays)

        # --- Move ---
        self.abs_pos += self.speed * self.direction

        # --- Check finish ---
        if (self.direction == 1 and self.abs_pos >= MAP_END) or \
           (self.direction == -1 and self.abs_pos <= 0):
            self.finished = True
            return

        self.abs_pos = max(0.0, min(float(MAP_END), self.abs_pos))

        # --- Stamp new position ---
        self.stamp(road_arrays)
        self.travel_time += 1


class GoalVehicle(Vehicle):
    """
    Interior-spawned vehicle following precomputed A* legs; finishes at route goal
    (not at map boundaries). path_xy is in simulation (x, y) metres for drawing.

    Each step, the caller passes min(gap ahead, stop-line distance); this class
    further limits by distance to the current leg end, so lights and car-following
    apply the same way as for regular vehicles.
    """

    def __init__(
        self,
        legs: List[DriveLeg],
        path_xy: List[Tuple[float, float]],
    ):
        if not legs:
            raise ValueError("GoalVehicle requires a non-empty leg list")
        first = legs[0]
        direction = 1 if first.end_abs > first.start_abs else (-1 if first.end_abs < first.start_abs else 1)
        super().__init__(
            "goal_car",
            first.axis,
            direction,
            first.channel_idx,
            abs_pos_override=first.start_abs,
        )
        self._legs = legs
        self._leg_i = 0
        self.path_xy = path_xy
        self.goal_xy = (float(path_xy[-1][0]), float(path_xy[-1][1])) if path_xy else (0.0, 0.0)

    def _dist_to_leg_end(self) -> float:
        leg = self._legs[self._leg_i]
        target = leg.end_abs
        if self.direction == 1:
            return max(0.0, target - self.abs_pos)
        return max(0.0, self.abs_pos - target)

    def _advance_leg(self, road_arrays) -> None:
        """Snap to leg end, skip zero-length legs, and move to next leg or finish."""
        leg = self._legs[self._leg_i]
        self.erase(road_arrays)
        self.abs_pos = leg.end_abs
        self.speed = 0.0

        if leg.is_final:
            self.finished = True
            return

        # Skip any zero-length legs (start == end) that can arise from coincident positions.
        while True:
            self._leg_i += 1
            if self._leg_i >= len(self._legs):
                self.finished = True
                return
            nxt = self._legs[self._leg_i]
            if abs(nxt.end_abs - nxt.start_abs) > 0.5:
                break
            if nxt.is_final:
                self.finished = True
                return

        self.axis = nxt.axis
        self.channel_idx = nxt.channel_idx
        self.abs_pos = nxt.start_abs
        if nxt.end_abs > nxt.start_abs:
            self.direction = 1
        elif nxt.end_abs < nxt.start_abs:
            self.direction = -1

        # Check that the new position is clear before stamping (prevents array corruption).
        si, li = pos_to_seg(self.abs_pos, self.axis)
        key = (self.axis, self.channel_idx, si, self.direction)
        arr = road_arrays.get(key)
        if arr is not None:
            li_c = max(0, min(int(li), len(arr) - 1))
            if arr[li_c] != 0:
                # Space is occupied — finish early rather than corrupt the array.
                self.finished = True
                return

        self.speed = float(self.max_speed)
        self.stamp(road_arrays)

    def step(self, road_arrays, dist_to_front: float):
        if self.finished:
            return

        d_leg = self._dist_to_leg_end()
        # dist_to_front = min(gap to vehicle ahead, stop-line distance); do not merge
        # d_leg into that before applying the bumper buffer (see _speed_goal_vehicle).
        self.speed, hold = _speed_goal_vehicle(
            dist_to_front, d_leg, self.speed, self.max_speed, self.accel
        )
        if hold:
            self.idling_time += 1

        if self.speed == 0:
            self.travel_time += 1
            return

        self.erase(road_arrays)
        self.abs_pos += self.speed * self.direction

        leg = self._legs[self._leg_i]
        target = leg.end_abs
        if self.direction == 1 and self.abs_pos >= target:
            self.abs_pos = target
            self.stamp(road_arrays)
            self.travel_time += 1
            self._advance_leg(road_arrays)
            return
        if self.direction == -1 and self.abs_pos <= target:
            self.abs_pos = target
            self.stamp(road_arrays)
            self.travel_time += 1
            self._advance_leg(road_arrays)
            return

        self.abs_pos = max(0.0, min(float(MAP_END), self.abs_pos))
        self.stamp(road_arrays)
        self.travel_time += 1


def _interior_spawn_clear(
    road_arrays: dict,
    axis: str,
    channel_idx: int,
    direction: int,
    front_abs: float,
    v_len: int,
) -> bool:
    """True if body cells and forward lookahead corridor are empty."""
    pos = front_abs
    for _ in range(v_len):
        if pos < 0 or pos > MAP_END:
            return False
        si, li = pos_to_seg(pos, axis)
        key = (axis, channel_idx, si, direction)
        arr = road_arrays.get(key)
        if arr is None:
            return False
        li_c = max(0, min(li, len(arr) - 1))
        if arr[li_c] != 0:
            return False
        pos -= direction

    look = v_len + SPEED_LIMIT + 2
    pos = front_abs
    for _ in range(look):
        if (direction == 1 and pos >= MAP_END) or (direction == -1 and pos <= 0):
            break
        si, li = pos_to_seg(pos, axis)
        key = (axis, channel_idx, si, direction)
        arr = road_arrays.get(key)
        if arr is None:
            return False
        li_c = max(0, min(li, len(arr) - 1))
        if arr[li_c] != 0:
            return False
        pos += direction
    return True


def try_spawn_one_vehicle(
    road_arrays: dict,
    vehicles: list,
    active_vehicles: list,
) -> None:
    """
    One spawn attempt under SPAWN_RATE (caller decides whether this runs).

    If MAX_ACTIVE_VEHICLES_FOR_SPAWN is set and the active list is already at
    that size, the attempt is a no-op (back-pressure).

    With probability uniform in [GOAL_VEHICLE_FRACTION_MIN, GOAL_VEHICLE_FRACTION_MAX],
    this attempt is reserved for a goal-directed vehicle: we retry planning/clearance
    several times; if all fail, the attempt is skipped (no fallback) so the share of
    goal cars stays near that probability when routes and space are available.

    Otherwise we use the legacy boundary spawner (car / bus / truck).
    """
    if MAX_ACTIVE_VEHICLES_FOR_SPAWN is not None and len(
        active_vehicles
    ) >= MAX_ACTIVE_VEHICLES_FOR_SPAWN:
        return

    p_goal = random.uniform(GOAL_VEHICLE_FRACTION_MIN, GOAL_VEHICLE_FRACTION_MAX)
    if random.random() < p_goal:
        ih = np.array(INTERSECTIONS_H, dtype=float)
        iv = np.array(INTERSECTIONS_V, dtype=float)
        for _ in range(8):
            planned = try_plan_goal_route(
                random,
                GRID_SIZE,
                ih,
                iv,
                float(MAP_END),
                max_sample_tries=25,
            )
            if planned is None:
                continue
            spawn, _goal, legs, path_xy = planned
            first = legs[0]
            direction = 1 if first.end_abs > first.start_abs else (-1 if first.end_abs < first.start_abs else 1)
            v_len = VTYPES["goal_car"]["length"]
            if _interior_spawn_clear(
                road_arrays,
                spawn.axis,
                spawn.channel_idx,
                direction,
                spawn.abs_pos,
                v_len,
            ):
                nv = GoalVehicle(legs, path_xy)
                nv.stamp(road_arrays)
                vehicles.append(nv)
                active_vehicles.append(nv)
                return
        return

    axis = random.choice(["h", "v"])
    direction = random.choice([1, -1])
    ch_idx = random.randint(0, GRID_SIZE - 1)
    vtype = random.choice(["car", "bus", "truck"])
    v_len = VTYPES[vtype]["length"]

    spawn_abs = 0.0 if direction == 1 else float(MAP_END)
    si, li = pos_to_seg(spawn_abs, axis)
    key = (axis, ch_idx, si, direction)
    arr = road_arrays.get(key)
    can_spawn = False
    if arr is not None:
        if direction == 1:
            check_cells = arr[: v_len + SPEED_LIMIT + 2]
        else:
            check_cells = arr[-(v_len + SPEED_LIMIT + 2) :]
        can_spawn = not np.any(check_cells != 0)

    if can_spawn:
        nv = Vehicle(vtype, axis, direction, ch_idx)
        nv.stamp(road_arrays)
        vehicles.append(nv)
        active_vehicles.append(nv)


# ---------------------------------------------------------------------------
# TimingBlock (same genotype as original)
# ---------------------------------------------------------------------------

class TimingBlock:
    """
    Encapsulates one traffic-light phase block.
    Enforces mandatory yellow (6s) + all-red (3s) safety transitions.
    """

    def __init__(self, g_ns: int, g_ew: int):
        self.g_ns = g_ns
        self.g_ew = g_ew
        self.seq = (
            ([0] * g_ns)
            + [1, 1, 1, 1, 1, 1, 4, 4, 4]
            + ([2] * g_ew)
            + [3, 3, 3, 3, 3, 3, 4, 4, 4]
        )

# ---------------------------------------------------------------------------
# Simulation engine
# ---------------------------------------------------------------------------

def _clear_arrays(road_arrays):
    """Zero out all road arrays (called at the start of each evaluation)."""
    for arr in road_arrays.values():
        arr[:] = 0


def _sort_vehicles_for_step(vehicles: list) -> None:
    """
    In-place sort so leaders are stepped before followers on each lane.

    The sim updates vehicles sequentially while mutating shared road arrays; if a
    follower runs before the car ahead moves, gap detection is pessimistic and
    platoons can freeze even when the light is green. Process front-to-back per
    (axis, channel, direction).
    """
    vehicles.sort(
        key=lambda v: (
            v.axis,
            v.channel_idx,
            v.direction,
            (-v.abs_pos if v.direction == 1 else v.abs_pos),
            v.id,
        )
    )


def _dist_ahead_in_arrays(vehicle: Vehicle, road_arrays: dict) -> float:
    """
    Scan forward in the road arrays from the vehicle's front and return the
    gap (in metres) to the nearest occupied cell.

    Uses numpy slicing per segment instead of a per-cell Python loop,
    which is significantly faster for the common case where the gap is
    within the current segment (~150–300 m) or the road is clear.

    Returns float('inf') if the road ahead is clear.
    """
    look = max(
        CAR_FOLLOW_LOOKAHEAD_M,
        int(vehicle.max_speed) + vehicle.length + 12,
    )
    vid = vehicle.id
    axis = vehicle.axis
    ch = vehicle.channel_idx
    d = vehicle.direction
    seg_lengths = SEGMENT_LENGTHS_V if axis == 'h' else SEGMENT_LENGTHS_H
    n_segs = len(seg_lengths)

    si, li = pos_to_seg(vehicle.abs_pos, axis)
    scanned = 0

    if d == 1:
        cursor = li + 1
        seg = si
        while scanned < look and 0 <= seg < n_segs:
            arr = road_arrays.get((axis, ch, seg, d))
            if arr is None:
                break
            slen = len(arr)
            if cursor >= slen:
                seg += 1
                cursor = 0
                continue
            end = min(slen, cursor + look - scanned)
            sl = arr[cursor:end]
            mask = (sl != 0) & (sl != vid)
            nz = np.flatnonzero(mask)
            if nz.size:
                return float(scanned + int(nz[0]) + 1)
            scanned += end - cursor
            seg += 1
            cursor = 0
    else:
        cursor = li - 1
        seg = si
        while scanned < look and 0 <= seg < n_segs:
            arr = road_arrays.get((axis, ch, seg, d))
            if arr is None:
                break
            if cursor < 0:
                seg -= 1
                if 0 <= seg < n_segs:
                    cursor = seg_lengths[seg] - 1
                continue
            start = max(0, cursor - (look - scanned) + 1)
            sl = arr[start:cursor + 1]
            if sl.size == 0:
                seg -= 1
                if 0 <= seg < n_segs:
                    cursor = seg_lengths[seg] - 1
                continue
            sl_rev = sl[::-1]
            mask = (sl_rev != 0) & (sl_rev != vid)
            nz = np.flatnonzero(mask)
            if nz.size:
                return float(scanned + int(nz[0]) + 1)
            scanned += cursor - start + 1
            seg -= 1
            if 0 <= seg < n_segs:
                cursor = seg_lengths[seg] - 1

    return float('inf')


def _dist_to_stop_line(vehicle: Vehicle, light: int) -> float:
    """
    Return the distance to the nearest upcoming red stop line.
    Returns float('inf') if the next light is green/yellow for this vehicle.
    """
    STOP_OFFSET = 12  # metres before intersection centre

    is_green = (
        (light == 0 or light == 1) if vehicle.axis == 'v'
        else (light == 2 or light == 3)
    )
    if is_green:
        return float('inf')

    valid_stops = []
    # h-vehicles travel along x and are stopped by vertical roads (INTERSECTIONS_V)
    # v-vehicles travel along y and are stopped by horizontal roads (INTERSECTIONS_H)
    stop_coords = INTERSECTIONS_V if vehicle.axis == 'h' else INTERSECTIONS_H
    for ix in stop_coords:
        stop_line = ix - STOP_OFFSET if vehicle.direction == 1 else ix + STOP_OFFSET
        if (vehicle.direction == 1 and vehicle.abs_pos < stop_line) or \
           (vehicle.direction == -1 and vehicle.abs_pos > stop_line):
            valid_stops.append(stop_line)

    if not valid_stops:
        return float('inf')

    if vehicle.direction == 1:
        nxt = min(valid_stops)
        return max(0.0, float(nxt - vehicle.abs_pos))
    nxt = max(valid_stops)
    return max(0.0, float(vehicle.abs_pos - nxt))


def evaluate(genes, seed, gen_idx=0, verbose=False):
    """
    Simulate one hour of traffic using physical road arrays.

    Returns the normalised fitness score (lower = better).
    Set verbose=True to print a live snapshot every 10 simulation-minutes.

    Evaluation validity (demand vs capacity)
    ----------------------------------------
    Expected spawn attempts per hour: 3600 * SPAWNS_PER_SECOND * SPAWN_RATE
    (each attempt may still fail: goal-slot retries, boundary full, optional
    MAX_ACTIVE_VEHICLES_FOR_SPAWN). If sustained injection exceeds what the
    network can discharge through exits + goal completions, active vehicles and
    idling *should* rise over time — that is a feature of the stress test, not
    necessarily a physics bug. Use MAX_ACTIVE_VEHICLES_FOR_SPAWN to meter input
    when you need a bounded queue for visualization or ablation studies.

    All individuals in a generation use the same ``seed`` so fitness comparisons
    are paired on identical traffic randomness (fair selection pressure).
    """
    random.seed(seed)
    Vehicle._next_id = 1

    road_arrays = build_road_arrays()

    vehicles: list[Vehicle] = []
    active_vehicles: list[Vehicle] = []

    # Flatten timing blocks into a 3600-second schedule
    flat = []
    for b in genes:
        flat.extend(b.seq)
    if len(flat) < 3600:
        flat.extend([4] * (3600 - len(flat)))
    schedule = np.array(flat[:3600], dtype=int)

    for t in range(3600):
        # --- Spawning: attempt SPAWNS_PER_SECOND times per tick ---
        for _ in range(SPAWNS_PER_SECOND):
            if random.random() < SPAWN_RATE:
                try_spawn_one_vehicle(road_arrays, vehicles, active_vehicles)

        light = int(schedule[t])

        # --- Physics step ---
        _sort_vehicles_for_step(active_vehicles)
        next_active = []
        for v in active_vehicles:
            dist_arr   = _dist_ahead_in_arrays(v, road_arrays)
            dist_light = _dist_to_stop_line(v, light)
            dist       = min(dist_arr, dist_light)

            v.step(road_arrays, dist)

            if not v.finished:
                next_active.append(v)

        active_vehicles = next_active

        # --- Verbose snapshot every 10 sim-minutes ---
        if verbose and (t + 1) % 600 == 0:
            finished_so_far = sum(1 for v in vehicles if v.finished)
            idling_now      = sum(1 for v in active_vehicles if v.speed == 0)
            print(f"  t={t+1:4d}s | spawned={len(vehicles):4d} | active={len(active_vehicles):4d} "
                  f"| finished={finished_so_far:4d} | idling={idling_now:3d} "
                  f"| light={light}")

    # --- Fitness ---
    # All vehicles (finished or not) contribute their accumulated travel and
    # idling time. Unfinished vehicles naturally accrue high scores because
    # their travel_time kept ticking and idling_time is weighted 2x.
    finished  = [v for v in vehicles if v.finished]
    unfinished = [v for v in vehicles if not v.finished]
    score = sum(v.travel_time + v.idling_time * 2 for v in vehicles)
    normalised = score / max(len(vehicles), 1)

    total_idle = sum(v.idling_time for v in vehicles)
    avg_tt     = sum(v.travel_time for v in finished) / max(len(finished), 1)

    if verbose:
        print(f"  --- Sim done | total spawned={len(vehicles)} | finished={len(finished)} "
              f"| unfinished={len(unfinished)} | total idling-secs={total_idle} "
              f"| avg travel time (finished)={avg_tt:.1f}s | score={normalised:.2f} ---")

    stats = {
        'spawned': len(vehicles),
        'finished': len(finished),
        'unfinished': len(unfinished),
        'total_idle': total_idle,
        'avg_tt': avg_tt,
    }
    return normalised, stats

# ---------------------------------------------------------------------------
# Evolutionary operators
# ---------------------------------------------------------------------------

def tournament_selection(population, fitness_scores, k=3):
    idx_pool = random.sample(range(len(population)), k)
    best     = min(idx_pool, key=lambda i: fitness_scores[i])
    return population[best]


def crossover(p1, p2):
    if random.random() < CROSSOVER_RATE:
        split = random.randint(1, len(p1) - 1)
        return p1[:split] + p2[split:]
    return copy.deepcopy(p1)


def project_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def runs_root() -> str:
    return os.path.join(project_dir(), "runs")


def create_timestamped_run_dir() -> str:
    """Create runs/YYYY-MM-DD_HHMMSS/ for this training run (see register_completed_run)."""
    root = runs_root()
    os.makedirs(root, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out = os.path.join(root, stamp)
    os.makedirs(out, exist_ok=True)
    return out


def register_completed_run(run_dir: str) -> None:
    """Point runs/latest_run.txt at this folder so validation/replay load the newest pickle."""
    stamp = os.path.basename(os.path.normpath(run_dir))
    latest = os.path.join(runs_root(), "latest_run.txt")
    os.makedirs(runs_root(), exist_ok=True)
    with open(latest, "w", encoding="utf-8") as f:
        f.write(stamp + "\n")


def read_latest_run_stamp() -> Optional[str]:
    p = os.path.join(runs_root(), "latest_run.txt")
    if not os.path.isfile(p):
        return None
    with open(p, encoding="utf-8") as f:
        s = f.read().strip()
    return s or None


def resolve_pkl_path(pkl_arg: Optional[str] = None) -> str:
    """Pickle path: explicit --pkl, else runs/<latest>/best_timing_array.pkl, else legacy cwd file."""
    if pkl_arg:
        return os.path.abspath(pkl_arg)
    stamp = read_latest_run_stamp()
    if stamp:
        cand = os.path.join(runs_root(), stamp, "best_timing_array.pkl")
        if os.path.isfile(cand):
            return cand
    return os.path.join(project_dir(), "best_timing_array.pkl")


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse
    import time as _time
    from concurrent.futures import ProcessPoolExecutor, as_completed

    parser = argparse.ArgumentParser(description="EA traffic-light training")
    parser.add_argument('-j', '--workers', type=int, default=0,
                        help='Parallel evaluation workers (0 = auto-detect CPU count)')
    parser.add_argument('--gens', type=int, default=GENS,
                        help=f'Number of generations (default {GENS})')
    parser.add_argument('--pop-size', type=int, default=POP_SIZE,
                        help=f'Population size (default {POP_SIZE})')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Re-run best individual with verbose mid-sim snapshots each gen')
    cli = parser.parse_args()

    n_workers = cli.workers or os.cpu_count() or 4
    n_gens    = cli.gens
    n_pop     = cli.pop_size

    run_dir = create_timestamped_run_dir()
    pkl_out = os.path.join(run_dir, "best_timing_array.pkl")
    print(f"Run output directory: {run_dir}")

    pop = [
        [TimingBlock(random.randint(15, 50), random.randint(15, 50)) for _ in range(40)]
        for _ in range(n_pop)
    ]
    baseline_genes = [TimingBlock(30, 30) for _ in range(60)]

    print(f"Starting EA | POP_SIZE={n_pop} | GENS={n_gens} "
          f"| MUTATION_RATE={MUTATION_RATE} | CROSSOVER_RATE={CROSSOVER_RATE}")
    print(f"Workers: {n_workers}  (parallel evaluation)")
    print(f"Grid: {GRID_SIZE}x{GRID_SIZE}")
    print(f"  H roads (y-coords): {[f'{x:.0f}' for x in INTERSECTIONS_H]}")
    print(f"  V roads (x-coords): {[f'{x:.0f}' for x in INTERSECTIONS_V]}")
    print(f"  H seg lengths: {SEGMENT_LENGTHS_H}")
    print(f"  V seg lengths: {SEGMENT_LENGTHS_V}")
    print("-" * 80)

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        for gen in range(n_gens):
            seed = 3000 + gen
            gen_t0 = _time.perf_counter()
            print(f"\n[Gen {gen:02d}/{n_gens-1}] Seed={seed}")

            # Submit baseline + all individuals concurrently
            baseline_future = pool.submit(evaluate, baseline_genes, seed)
            future_to_idx = {}
            for i, ind in enumerate(pop):
                future_to_idx[pool.submit(evaluate, ind, seed, gen)] = i

            base_score, _ = baseline_future.result()

            scores      = [0.0] * n_pop
            stats_list  = [None] * n_pop
            done = 0
            for f in as_completed(future_to_idx):
                idx = future_to_idx[f]
                sc, st = f.result()
                scores[idx] = sc
                stats_list[idx] = st
                done += 1
                sys.stdout.write(f"\r  Evaluating: {done}/{n_pop} done")
                sys.stdout.flush()
            sys.stdout.write("\r" + " " * 50 + "\r")
            sys.stdout.flush()

            scores_arr  = np.array(scores)
            best_idx    = int(np.argmin(scores_arr))
            best_score  = scores_arr[best_idx]
            mean_score  = scores_arr.mean()
            worst_score = scores_arr.max()
            diff        = base_score - best_score

            print(f"  Scores  -> best: {best_score:.2f}  mean: {mean_score:.2f}  worst: {worst_score:.2f}")
            print(f"  Baseline-> {base_score:.2f}  |  EA improvement over baseline: {diff:+.2f}")

            bs = stats_list[best_idx]
            print(f"  Best #{best_idx}: spawned={bs['spawned']} finished={bs['finished']} "
                  f"unfinished={bs['unfinished']} idle={bs['total_idle']}s "
                  f"avg_tt={bs['avg_tt']:.1f}s")

            if cli.verbose:
                print(f"  Verbose re-evaluation of best individual:")
                evaluate(pop[best_idx], seed, gen_idx=gen, verbose=True)

            gen_elapsed = _time.perf_counter() - gen_t0
            print(f"  Generation wall time: {gen_elapsed:.1f}s")

            # Elitism + reproduction
            new_pop = [pop[best_idx]]
            while len(new_pop) < n_pop:
                p1    = tournament_selection(pop, scores, k=TOURNAMENT_K)
                p2    = tournament_selection(pop, scores, k=TOURNAMENT_K)
                child = crossover(p1, p2)
                if random.random() < MUTATION_RATE:
                    child[random.randint(0, len(child) - 1)] = TimingBlock(
                        random.randint(10, 80), random.randint(10, 80)
                    )
                new_pop.append(child)

            pop = new_pop
            print("-" * 80)

    with open(pkl_out, 'wb') as f:
        pickle.dump(pop[0], f)
    register_completed_run(run_dir)

    print(f"\nTraining complete. Best strategy saved to:\n  {pkl_out}")
