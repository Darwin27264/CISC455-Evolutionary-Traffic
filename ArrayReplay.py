"""
ArrayReplay.py
--------------
Pygame visualiser for the best strategy saved in best_timing_array.pkl.

Replay uses lower spawn intensity than training (see REPLAY_SPAWN_RATE_MULT) so the map
stays readable. Goal cars get a unique color; the matching line is the remaining route
and the diamond is the destination.

Controls:
  SPACE  — pause / unpause
  +/-    — speed up / slow down
  Q/Esc  — quit
"""

import pickle
import random
import sys
import os
import colorsys
import pygame

sys.path.insert(0, os.path.dirname(__file__))


def _pkl_from_argv():
    args = sys.argv[1:]
    for i, a in enumerate(args):
        if a == '--pkl' and i + 1 < len(args):
            return args[i + 1]
    return None


from ArrayBasedTraining import (
    Vehicle,
    GoalVehicle,
    TimingBlock,
    VTYPES,
    INTERSECTIONS_H,
    INTERSECTIONS_V,
    MAP_END,
    GRID_SIZE,
    SPAWN_RATE,
    SPAWNS_PER_SECOND,
    SPEED_LIMIT,
    build_road_arrays,
    _dist_ahead_in_arrays,
    _dist_to_stop_line,
    _sort_vehicles_for_step,
    try_spawn_one_vehicle,
    resolve_pkl_path,
)
import numpy as np

# ---------------------------------------------------------------------------
# Display settings
# ---------------------------------------------------------------------------
WIN_W, WIN_H = 900, 950   # extra 150px at bottom for HUD
SIM_W, SIM_H = 900, 800   # simulation area
SCALE = SIM_W / MAP_END   # metres → pixels  (0.9 px/m for 1000m map)

# Replay-only: lighter traffic so the window stays readable (training / validation unchanged).
REPLAY_SPAWN_RATE_MULT = 0.40
REPLAY_SPAWNS_PER_TICK = max(1, int(round(SPAWNS_PER_SECOND * 0.45)))

ROAD_HALF_W  = 18          # pixels, half the road width drawn on screen
LANE_OFFSET  = 6           # pixels, offset from road centre for each direction lane
STOP_MARK_LEN = 6          # pixels, length of the stop-line tick

VEHICLE_COLORS = {
    'car':   (70,  130, 255),
    'bus':   (255, 160,  20),
    'truck': (220,  60,  60),
    'goal_car': (0, 255, 200),  # fallback if route color missing
}
LIGHT_COLORS = {
    0: (0,   220,   0),   # NS green
    1: (255, 220,   0),   # NS yellow
    2: (0,   220,   0),   # EW green  (same dot colour, direction tells you which)
    3: (255, 220,   0),   # EW yellow
    4: (200,   0,   0),   # all red
}

# ---------------------------------------------------------------------------
# Load the saved chromosome
# ---------------------------------------------------------------------------
PKL = os.path.abspath(resolve_pkl_path(_pkl_from_argv()))
with open(PKL, 'rb') as f:
    best_blocks = pickle.load(f)

flat = []
for b in best_blocks:
    flat.extend(b.seq)
if len(flat) < 3600:
    flat.extend([4] * (3600 - len(flat)))
schedule = np.array(flat[:3600], dtype=int)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def to_px(metres):
    return int(metres * SCALE)


def goal_route_color(vehicle_id: int) -> tuple:
    """Stable saturated RGB per vehicle id so car, path, and destination match."""
    h = (vehicle_id * 0.618033988749895) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.90, 0.98)
    return (int(r * 255), int(g * 255), int(b * 255))


def vehicle_sim_xy_m(v) -> tuple:
    """Simulation (x, y) in metres — same frame as path_xy."""
    if v.axis == "h":
        return (float(v.abs_pos), float(INTERSECTIONS_H[v.channel_idx]))
    return (float(INTERSECTIONS_V[v.channel_idx]), float(v.abs_pos))


def remaining_route_polyline(v, path_xy: list) -> list:
    """From the car's current position along the planned route to the goal."""
    if len(path_xy) < 2:
        return list(path_xy)
    cx, cy = vehicle_sim_xy_m(v)
    best_i = 0
    best_d = 1e18
    for i, (px, py) in enumerate(path_xy):
        d = (px - cx) ** 2 + (py - cy) ** 2
        if d < best_d:
            best_d = d
            best_i = i
    tail = path_xy[best_i:]
    # If we're essentially at tail[0], skip duplicate vertex
    if tail and (tail[0][0] - cx) ** 2 + (tail[0][1] - cy) ** 2 < 64.0:
        tail = tail[1:]
    return [(cx, cy)] + [tuple(p) for p in tail]


def draw_road_network(surf):
    """Draw the static road grid once per frame (background layer)."""
    surf.fill((180, 180, 180))

    ROAD_COL = (60, 60, 60)
    DASH_COL = (200, 200, 80)
    KERB_COL = (100, 100, 100)

    boundaries_h_px = [0] + [to_px(ix) for ix in INTERSECTIONS_V] + [SIM_W]
    boundaries_v_px = [0] + [to_px(ix) for ix in INTERSECTIONS_H] + [SIM_H]

    # Horizontal road segments
    for ch_idx in range(GRID_SIZE):
        cy = to_px(INTERSECTIONS_H[ch_idx])
        for seg in range(len(boundaries_h_px) - 1):
            x0, x1 = boundaries_h_px[seg], boundaries_h_px[seg + 1]
            seg_x0 = x0 + (ROAD_HALF_W if seg > 0 else 0)
            seg_x1 = x1 - (ROAD_HALF_W if seg < len(boundaries_h_px) - 2 else 0)
            if seg_x1 <= seg_x0:
                continue
            pygame.draw.rect(surf, ROAD_COL, (seg_x0, cy - ROAD_HALF_W, seg_x1 - seg_x0, ROAD_HALF_W * 2))
            pygame.draw.line(surf, KERB_COL, (seg_x0, cy - ROAD_HALF_W), (seg_x1, cy - ROAD_HALF_W), 1)
            pygame.draw.line(surf, KERB_COL, (seg_x0, cy + ROAD_HALF_W), (seg_x1, cy + ROAD_HALF_W), 1)
            for dx in range(seg_x0, seg_x1, 20):
                pygame.draw.line(surf, DASH_COL, (dx, cy), (min(dx + 10, seg_x1), cy), 1)

    # Vertical road segments
    for ch_idx in range(GRID_SIZE):
        cx = to_px(INTERSECTIONS_V[ch_idx])
        for seg in range(len(boundaries_v_px) - 1):
            y0, y1 = boundaries_v_px[seg], boundaries_v_px[seg + 1]
            seg_y0 = y0 + (ROAD_HALF_W if seg > 0 else 0)
            seg_y1 = y1 - (ROAD_HALF_W if seg < len(boundaries_v_px) - 2 else 0)
            if seg_y1 <= seg_y0:
                continue
            pygame.draw.rect(surf, ROAD_COL, (cx - ROAD_HALF_W, seg_y0, ROAD_HALF_W * 2, seg_y1 - seg_y0))
            pygame.draw.line(surf, KERB_COL, (cx - ROAD_HALF_W, seg_y0), (cx - ROAD_HALF_W, seg_y1), 1)
            pygame.draw.line(surf, KERB_COL, (cx + ROAD_HALF_W, seg_y0), (cx + ROAD_HALF_W, seg_y1), 1)
            for dy in range(seg_y0, seg_y1, 20):
                pygame.draw.line(surf, DASH_COL, (cx, dy), (cx, min(dy + 10, seg_y1)), 1)

    # Intersection boxes
    for rx_idx in range(GRID_SIZE):
        for ry_idx in range(GRID_SIZE):
            cx = to_px(INTERSECTIONS_V[rx_idx])
            cy = to_px(INTERSECTIONS_H[ry_idx])
            pygame.draw.rect(surf, (45, 45, 45),
                             (cx - ROAD_HALF_W, cy - ROAD_HALF_W, ROAD_HALF_W * 2, ROAD_HALF_W * 2))

    # Segment length labels
    label_font = pygame.font.SysFont("Consolas", 11)
    boundaries_h_m = [0] + list(INTERSECTIONS_V) + [MAP_END]
    cy0 = to_px(INTERSECTIONS_H[0])
    for i in range(len(boundaries_h_m) - 1):
        seg_len_m = int(boundaries_h_m[i + 1] - boundaries_h_m[i])
        mid_px = (boundaries_h_px[i] + boundaries_h_px[i + 1]) // 2
        lbl = label_font.render(f"{seg_len_m}m", True, (220, 220, 100))
        surf.blit(lbl, (mid_px - lbl.get_width() // 2, cy0 - ROAD_HALF_W - 14))


def draw_lights(surf, light):
    """Draw a coloured dot at every intersection for the current light state."""
    for rx_idx in range(GRID_SIZE):
        for ry_idx in range(GRID_SIZE):
            cx = to_px(INTERSECTIONS_V[rx_idx])
            cy = to_px(INTERSECTIONS_H[ry_idx])
            ns_c = (0, 220, 0) if light in (0, 1) else (200, 0, 0)
            ew_c = (0, 220, 0) if light in (2, 3) else (200, 0, 0)
            if light == 1: ns_c = (255, 220, 0)
            if light == 3: ew_c = (255, 220, 0)
            pygame.draw.circle(surf, ns_c, (cx, cy - ROAD_HALF_W - 8), 7)
            pygame.draw.circle(surf, ew_c, (cx - ROAD_HALF_W - 8, cy), 7)


def draw_goal_routes_and_destinations(surf, vehicles):
    """
    For each goal car: draw remaining route + destination marker.

    Lines are drawn on an alpha overlay only — they never interact with simulation
    physics (collision uses road_arrays in ArrayBasedTraining, not pixels).
    """
    overlay = pygame.Surface((SIM_W, SIM_H), pygame.SRCALPHA)
    for v in vehicles:
        if v.finished or not isinstance(v, GoalVehicle):
            continue
        col = goal_route_color(v.id)
        line_pts = remaining_route_polyline(v, v.path_xy)
        if len(line_pts) >= 2:
            px_pts = [(to_px(x), to_px(y)) for x, y in line_pts]
            pygame.draw.lines(overlay, (20, 20, 20, 100), False, px_pts, 6)
            pygame.draw.lines(overlay, (*col, 200), False, px_pts, 3)
    surf.blit(overlay, (0, 0))

    for v in vehicles:
        if v.finished or not isinstance(v, GoalVehicle):
            continue
        col = goal_route_color(v.id)
        gx, gy = v.goal_xy
        gpx, gpy = to_px(gx), to_px(gy)
        s = 7
        pts_d = [(gpx, gpy - s), (gpx + s, gpy), (gpx, gpy + s), (gpx - s, gpy)]
        pygame.draw.polygon(surf, col, pts_d)
        pygame.draw.polygon(surf, (255, 255, 255), pts_d, 2)


def draw_vehicles(surf, vehicles):
    """Draw every active vehicle as a small rectangle on its lane."""
    for v in vehicles:
        if v.finished:
            continue
        if isinstance(v, GoalVehicle):
            col = goal_route_color(v.id)
        else:
            col = VEHICLE_COLORS.get(v.vtype, (150, 150, 150))

        front_px = to_px(v.abs_pos)
        rear_px  = to_px(v.abs_pos - v.direction * v.length)
        lo_px    = min(front_px, rear_px)
        v_len    = max(3, abs(front_px - rear_px))

        if v.axis == 'h':
            cy     = to_px(INTERSECTIONS_H[v.channel_idx])
            lane_y = cy - LANE_OFFSET - 5 if v.direction == 1 else cy + LANE_OFFSET
            pygame.draw.rect(surf, col, (lo_px, lane_y, v_len, 8))
            pygame.draw.rect(surf, (0, 0, 0), (lo_px, lane_y, v_len, 8), 1)
            pygame.draw.line(surf, (255, 255, 255),
                             (front_px, lane_y), (front_px, lane_y + 7), 2)
        else:
            cx     = to_px(INTERSECTIONS_V[v.channel_idx])
            lane_x = cx - LANE_OFFSET - 5 if v.direction == 1 else cx + LANE_OFFSET
            pygame.draw.rect(surf, col, (lane_x, lo_px, 8, v_len))
            pygame.draw.rect(surf, (0, 0, 0), (lane_x, lo_px, 8, v_len), 1)
            pygame.draw.line(surf, (255, 255, 255),
                             (lane_x, front_px), (lane_x + 7, front_px), 2)


def draw_hud(surf, font, small_font, t, light, vehicles, active, paused, speed_mult):
    """Draw the bottom HUD panel."""
    pygame.draw.rect(surf, (30, 30, 30), (0, SIM_H, WIN_W, WIN_H - SIM_H))

    finished = sum(1 for v in vehicles if v.finished)
    idling   = sum(1 for v in active   if v.speed == 0)

    state_names = {0: 'NS Green / EW Red', 1: 'NS Yellow / EW Red',
                   2: 'NS Red / EW Green', 3: 'NS Red / EW Yellow', 4: 'All Red'}

    mins, secs = divmod(t, 60)
    lines = [
        (f"Time: {mins:02d}:{secs:02d}  (t={t}/3600)",          (255, 255, 255)),
        (f"Light: {state_names.get(light, '?')}",                LIGHT_COLORS.get(light, (200,200,200))),
    ]
    if light == 4:
        lines.append(
            ("All approaches must stop (yellow→red clearance) — not blocked by route lines.",
             (255, 190, 160)),
        )
    lines.extend(
        [
            (f"Spawned: {len(vehicles)}  Active: {len(active)}  "
             f"Finished: {finished}  Idling: {idling}", (200, 200, 200)),
            (f"Speed: {speed_mult}x   [SPACE] pause  [+/-] speed  [Q] quit",
             (150, 150, 150)),
            ("Routes: visual only (no effect on traffic).", (130, 200, 130)),
        ]
    )
    if paused:
        lines.insert(0, ("--- PAUSED ---", (255, 200, 0)))

    for i, (txt, col) in enumerate(lines):
        f = font if txt.startswith("Time:") or txt.startswith("Light:") else small_font
        surf.blit(f.render(txt, True, col), (14, SIM_H + 8 + i * 20))

    # Colour legend (goal cars use unique colors — shown as swatch + note)
    legend = [("car", VEHICLE_COLORS["car"]), ("bus", VEHICLE_COLORS["bus"]),
              ("truck", VEHICLE_COLORS["truck"]), ("goal_car*", (180, 180, 180))]
    for i, (vt, col) in enumerate(legend):
        lx = WIN_W - 200
        ly = SIM_H + 10 + i * 20
        pygame.draw.rect(surf, col, (lx, ly + 2, 16, 10))
        pygame.draw.rect(surf, (80, 80, 80), (lx, ly + 2, 16, 10), 1)
        surf.blit(small_font.render(vt, True, (200, 200, 200)), (lx + 22, ly))
    surf.blit(small_font.render("*unique color + diamond = dest", True, (160, 160, 160)),
              (WIN_W - 200, SIM_H + 10 + len(legend) * 20))

# ---------------------------------------------------------------------------
# Main simulation loop
# ---------------------------------------------------------------------------

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("Array-Based Traffic Replay — Best Strategy")
    clock = pygame.time.Clock()

    font       = pygame.font.SysFont("Consolas", 17)
    small_font = pygame.font.SysFont("Consolas", 14)

    SEED = 3000
    random.seed(SEED)
    Vehicle._next_id = 1
    road_arrays    = build_road_arrays()
    vehicles       = []
    active_vehicles = []

    paused     = False
    speed_mult = 1      # render every Nth step (1 = real time, 2 = 2x, etc.)
    t          = 0

    running = True
    while running and t < 3600:

        # --- Events ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    speed_mult = min(speed_mult + 1, 20)
                elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    speed_mult = max(speed_mult - 1, 1)

        if paused:
            draw_road_network(screen)
            draw_goal_routes_and_destinations(screen, vehicles)
            draw_vehicles(screen, vehicles)
            draw_lights(screen, int(schedule[min(t, 3599)]))
            draw_hud(screen, font, small_font, t, int(schedule[min(t, 3599)]),
                     vehicles, active_vehicles, paused, speed_mult)
            pygame.display.flip()
            clock.tick(30)
            continue

        # --- Advance `speed_mult` simulation steps before re-drawing ---
        for _ in range(speed_mult):
            if t >= 3600:
                break

            for _ in range(REPLAY_SPAWNS_PER_TICK):
                if random.random() < SPAWN_RATE * REPLAY_SPAWN_RATE_MULT:
                    try_spawn_one_vehicle(road_arrays, vehicles, active_vehicles)

            light = int(schedule[t])
            _sort_vehicles_for_step(active_vehicles)
            next_active = []
            for v in active_vehicles:
                dist = min(_dist_ahead_in_arrays(v, road_arrays),
                           _dist_to_stop_line(v, light))
                v.step(road_arrays, dist)
                if not v.finished:
                    next_active.append(v)
            active_vehicles = next_active
            t += 1

        # --- Draw ---
        draw_road_network(screen)
        draw_goal_routes_and_destinations(screen, vehicles)
        draw_vehicles(screen, vehicles)
        draw_lights(screen, int(schedule[min(t - 1, 3599)]))
        draw_hud(screen, font, small_font, t, int(schedule[min(t - 1, 3599)]),
                 vehicles, active_vehicles, paused, speed_mult)
        pygame.display.flip()
        clock.tick(60)

    # --- End screen ---
    finished   = [v for v in vehicles if v.finished]
    unfinished = [v for v in vehicles if not v.finished]
    score      = sum(v.travel_time + v.idling_time * 2 for v in vehicles)

    screen.fill((20, 20, 20))
    lines = [
        "=== SIMULATION COMPLETE ===",
        f"Total spawned  : {len(vehicles)}",
        f"Finished       : {len(finished)} ({100*len(finished)/max(len(vehicles),1):.1f}%)",
        f"Unfinished     : {len(unfinished)}",
        f"Fitness score  : {score / max(len(vehicles), 1):.2f}",
        "",
        "Press Q or close window to exit.",
    ]
    big = pygame.font.SysFont("Consolas", 20)
    for i, line in enumerate(lines):
        screen.blit(big.render(line, True, (220, 220, 220)), (60, 80 + i * 34))
    pygame.display.flip()

    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                waiting = False
            elif event.type == pygame.KEYDOWN and event.key in (pygame.K_q, pygame.K_ESCAPE):
                waiting = False
        clock.tick(30)

    pygame.quit()


if __name__ == '__main__':
    main()
