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

import numpy as np
import random
import pickle
import copy
import sys

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
GRID_SIZE         = 4     # NxN intersections
HEADLESS_MODE     = True  # No pygame dependency for this script
SPAWN_RATE        = 0.25  # Probability per attempt of a vehicle spawning
SPAWNS_PER_SECOND = 4     # How many spawn attempts per simulation second
SPEED_LIMIT       = 10    # metres per second

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
    intersections = INTERSECTIONS_V if axis == 'h' else INTERSECTIONS_H
    seg_lengths   = SEGMENT_LENGTHS_V if axis == 'h' else SEGMENT_LENGTHS_H
    boundaries = [0] + intersections + [MAP_END]
    for i in range(len(boundaries) - 1):
        lo, hi = boundaries[i], boundaries[i + 1]
        if lo <= abs_pos < hi:
            return i, int(abs_pos - lo)
    last = len(seg_lengths) - 1
    return last, seg_lengths[last] - 1

def seg_to_pos(seg_idx, local_idx, axis='h'):
    """Inverse of pos_to_seg."""
    starts = SEGMENT_STARTS_V if axis == 'h' else SEGMENT_STARTS_H
    return starts[seg_idx] + local_idx

# ---------------------------------------------------------------------------
# Vehicle
# ---------------------------------------------------------------------------

# Vehicle type parameters
VTYPES = {
    'car':   {'length': 1, 'accel': 2, 'max_speed': SPEED_LIMIT},
    'bus':   {'length': 3, 'accel': 1, 'max_speed': SPEED_LIMIT},
    'truck': {'length': 5, 'accel': 1, 'max_speed': SPEED_LIMIT},
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

    def __init__(self, vtype: str, axis: str, direction: int, channel_idx: int):
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

        # Spawn at the boundary and at top speed
        self.abs_pos = 0.0 if direction == 1 else float(MAP_END)
        self.speed   = float(self.max_speed)

        self.travel_time = 0
        self.idling_time = 0
        self.finished    = False

    # ------------------------------------------------------------------
    # Array footprint helpers
    # ------------------------------------------------------------------

    def _write_to_arrays(self, road_arrays, value):
        """
        Write `value` into every cell occupied by this vehicle.
        The front of the vehicle is at abs_pos; each subsequent body cell
        trails one metre behind (opposite to direction of travel), so a
        vehicle of length L fills cells at:
            abs_pos, abs_pos - direction, abs_pos - 2*direction, …
        """
        pos = self.abs_pos
        for _ in range(self.length):
            si, li = pos_to_seg(pos, self.axis)
            key = (self.axis, self.channel_idx, si, self.direction)
            if key in road_arrays:
                arr = road_arrays[key]
                li_clamped = max(0, min(li, len(arr) - 1))
                arr[li_clamped] = value
            pos -= self.direction  # next body cell is one metre behind the front

    def stamp(self, road_arrays):
        """Stamp vehicle id into road arrays."""
        self._write_to_arrays(road_arrays, self.id)

    def erase(self, road_arrays):
        """Clear vehicle footprint from road arrays."""
        self._write_to_arrays(road_arrays, 0)

    # ------------------------------------------------------------------
    # Movement
    # ------------------------------------------------------------------

    def step(self, road_arrays, dist_to_front: float):
        """Advance vehicle by one second."""
        if self.finished:
            return

        # --- Acceleration / braking ---
        safety_buffer = 2
        if dist_to_front <= self.speed + safety_buffer:
            self.speed = max(0.0, dist_to_front - safety_buffer)
            if self.speed == 0:
                self.idling_time += 1
        else:
            self.speed = min(float(self.max_speed), self.speed + self.accel)

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


def _dist_ahead_in_arrays(vehicle: Vehicle, road_arrays: dict) -> float:
    """
    Scan forward in the road arrays from the vehicle's front and return the
    gap (in metres) to the nearest occupied cell.

    Returns float('inf') if the road ahead is clear.
    """
    look_ahead = int(vehicle.max_speed) + vehicle.length + 4  # metres to scan
    gap = float('inf')

    pos = vehicle.abs_pos + vehicle.direction  # start one metre ahead
    for step in range(look_ahead):
        if (vehicle.direction == 1 and pos >= MAP_END) or \
           (vehicle.direction == -1 and pos <= 0):
            break
        si, li = pos_to_seg(pos, vehicle.axis)
        key = (vehicle.axis, vehicle.channel_idx, si, vehicle.direction)
        if key in road_arrays:
            arr = road_arrays[key]
            li_c = max(0, min(li, len(arr) - 1))
            if arr[li_c] != 0 and arr[li_c] != vehicle.id:
                gap = step + 1  # distance is the step count (1-based)
                break
        pos += vehicle.direction

    return gap


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

    closest = min(valid_stops, key=lambda x: abs(x - vehicle.abs_pos))
    return abs(closest - vehicle.abs_pos)


def evaluate(genes, seed, gen_idx=0, verbose=False):
    """
    Simulate one hour of traffic using physical road arrays.

    Returns the normalised fitness score (lower = better).
    Set verbose=True to print a live snapshot every 10 simulation-minutes.
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
                axis      = random.choice(['h', 'v'])
                direction = random.choice([1, -1])
                ch_idx    = random.randint(0, GRID_SIZE - 1)
                vtype     = random.choice(['car', 'bus', 'truck'])
                v_len     = VTYPES[vtype]['length']

                # The spawn point is the boundary cell (0 for dir=1, MAP_END-1 for dir=-1)
                spawn_abs = 0.0 if direction == 1 else float(MAP_END)
                si, li = pos_to_seg(spawn_abs, axis)
                key = (axis, ch_idx, si, direction)

                # Check that enough space exists in the array for the vehicle body
                arr = road_arrays.get(key)
                can_spawn = False
                if arr is not None:
                    if direction == 1:
                        check_cells = arr[:v_len + SPEED_LIMIT + 2]
                    else:
                        check_cells = arr[-(v_len + SPEED_LIMIT + 2):]
                    can_spawn = not np.any(check_cells != 0)

                if can_spawn:
                    nv = Vehicle(vtype, axis, direction, ch_idx)
                    nv.stamp(road_arrays)
                    vehicles.append(nv)
                    active_vehicles.append(nv)

        light = int(schedule[t])

        # --- Physics step ---
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

    if verbose:
        total_idle = sum(v.idling_time for v in vehicles)
        avg_tt     = sum(v.travel_time for v in finished) / max(len(finished), 1)
        print(f"  --- Sim done | total spawned={len(vehicles)} | finished={len(finished)} "
              f"| unfinished={len(unfinished)} | total idling-secs={total_idle} "
              f"| avg travel time (finished)={avg_tt:.1f}s | score={normalised:.2f} ---")

    return normalised

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

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    # Initial population: each individual is a list of 40 TimingBlocks
    pop = [
        [TimingBlock(random.randint(15, 50), random.randint(15, 50)) for _ in range(40)]
        for _ in range(POP_SIZE)
    ]
    baseline_genes = [TimingBlock(30, 30) for _ in range(60)]

    print(f"Starting EA | POP_SIZE={POP_SIZE} | GENS={GENS} | MUTATION_RATE={MUTATION_RATE} | CROSSOVER_RATE={CROSSOVER_RATE}")
    print(f"Grid: {GRID_SIZE}x{GRID_SIZE}")
    print(f"  H roads (y-coords): {[f'{x:.0f}' for x in INTERSECTIONS_H]}")
    print(f"  V roads (x-coords): {[f'{x:.0f}' for x in INTERSECTIONS_V]}")
    print(f"  H seg lengths: {SEGMENT_LENGTHS_H}")
    print(f"  V seg lengths: {SEGMENT_LENGTHS_V}")
    print("-" * 80)

    for gen in range(GENS):
        seed = 3000 + gen
        print(f"\n[Gen {gen:02d}/{GENS-1}] Seed={seed}")

        sys.stdout.write(f"  Evaluating baseline...")
        sys.stdout.flush()
        base_score = evaluate(baseline_genes, seed)
        print(f"\r  Baseline score: {base_score:.2f}")

        scores = []
        for i, ind in enumerate(pop):
            sys.stdout.write(f"\r  Evaluating individual {i+1:3d}/{POP_SIZE}...")
            sys.stdout.flush()
            scores.append(evaluate(ind, seed, gen_idx=gen))
        sys.stdout.write("\r" + " " * 50 + "\r")
        sys.stdout.flush()

        scores_arr = np.array(scores)
        best_idx   = int(np.argmin(scores_arr))
        best_score = scores_arr[best_idx]
        mean_score = scores_arr.mean()
        worst_score= scores_arr.max()
        diff       = base_score - best_score

        print(f"  Scores  → best: {best_score:.2f}  mean: {mean_score:.2f}  worst: {worst_score:.2f}")
        print(f"  Baseline→ {base_score:.2f}  |  EA improvement over baseline: {diff:+.2f}")

        # Verbose breakdown of the best individual this generation
        print(f"  Best individual (idx={best_idx}) sim breakdown:")
        evaluate(pop[best_idx], seed, gen_idx=gen, verbose=True)

        # Elitism + reproduction
        new_pop = [pop[best_idx]]
        while len(new_pop) < POP_SIZE:
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

    with open('best_timing_array.pkl', 'wb') as f:
        pickle.dump(pop[0], f)

    print("\nTraining complete. Best strategy saved to best_timing_array.pkl")
