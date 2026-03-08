import numpy as np
import time
import heapq
import matplotlib.pyplot as plt
import pybullet as p
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.utils import sync

# ----- Simulation parameters -----
DRONE = DroneModel("cf2x")
NUM_DRONES = 1
PHYSICS = Physics("pyb")
GUI = True
OBSTACLES = False  # Disable built-in random obstacles; we'll spawn a custom maze
SIM_FREQ = 240   # Higher sim freq for smoother physics (was 120)
CTRL_FREQ = 48   # Higher control freq for more stable response (was 24)
DURATION_SEC = 180  # longer run to allow extinguishing multiple fires
TOTAL_FIRES = 3    # number of fires to extinguish before declaring success

# ----- Initial drone state -----
FLIGHT_HEIGHT = 0.5   # Altitude in m; avoid low hover (e.g. 0.1) for stability
INIT_XYZS = np.array([[0, 0, FLIGHT_HEIGHT]])
INIT_RPYS = np.array([[0, 0, 0]])

# ----- Occupancy map setup (6×6 m box, same as arena) -----
MAP_SIZE = 6
MAP_RES = 0.1
MAP_DIM = int(MAP_SIZE / MAP_RES)
occupancy_map = np.zeros((MAP_DIM, MAP_DIM), dtype=np.uint8)

FIRE_BODY_ID = None
FIRE_POS = None  # world (x, y, z) of the fire object
EXTINGUISHED_FIRE_XY = []  # (x, y) of each extinguished fire for final visualization

PILLAR_POSITIONS = [
    (-1.2, 1.2),
    (1.5, -0.8),
    (-0.8, -1.5),
    (1.8, 1.5),
    (-1.8, 0.0),
    (0.6, 0.8),
    (-0.6, -0.9),
    (1.0, 0.4),
    (-1.2, 0.6),
    (0.0, -1.8),
    (2.2, 0.0),
    (-2.2, -1.2),
    (0.9, -1.2),
    (-1.5, -0.5),
    (1.4, 0.9),
    (-0.3, 1.8),
    (2.0, -1.5),
]

def world_to_map(x, y):
    mx = int((x + MAP_SIZE/2)/MAP_RES)
    my = int((y + MAP_SIZE/2)/MAP_RES)
    return np.clip(mx, 0, MAP_DIM-1), np.clip(my, 0, MAP_DIM-1)

def map_to_world(mx, my):
    """Convert map indices back to world (x, y) coordinates."""
    x = (mx + 0.5) * MAP_RES - MAP_SIZE / 2.0
    y = (my + 0.5) * MAP_RES - MAP_SIZE / 2.0
    return x, y


def create_maze(py_client):
    """Spawn a 6×6 m box boundary and a few pillar obstacles in PyBullet."""
    wall_height = 1.0
    wall_thickness = 0.03
    half_h = wall_height / 2.0
    color = [0.2, 0.2, 0.8, 1.0]

    # Half-length of each wall for 6×6 m inner area (walls at ±3 m)
    box_half = 3.0

    def add_wall(x, y, half_length, orientation="x"):
        if orientation == "x":
            half_extents = [half_length, wall_thickness / 2.0, half_h]
        else:
            half_extents = [wall_thickness / 2.0, half_length, half_h]
        col = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=half_extents,
            physicsClientId=py_client,
        )
        vis = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=half_extents,
            rgbaColor=color,
            physicsClientId=py_client,
        )
        p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=[x, y, half_h],
            physicsClientId=py_client,
        )

    # Outer 6×6 m box boundary
    add_wall(0.0, box_half, half_length=box_half, orientation="x")    # top
    add_wall(0.0, -box_half, half_length=box_half, orientation="x")    # bottom
    add_wall(-box_half, 0.0, half_length=box_half, orientation="y")    # left
    add_wall(box_half, 0.0, half_length=box_half, orientation="y")     # right

    # Pillar obstacles (cylinders, same height as walls)
    pillar_radius = 0.08
    pillar_color = [0.4, 0.3, 0.2, 1.0]
    col_pillar = p.createCollisionShape(
        p.GEOM_CYLINDER,
        radius=pillar_radius,
        height=wall_height,
        physicsClientId=py_client,
    )
    vis_pillar = p.createVisualShape(
        p.GEOM_CYLINDER,
        radius=pillar_radius,
        length=wall_height,
        rgbaColor=pillar_color,
        physicsClientId=py_client,
    )
    for px, py in PILLAR_POSITIONS:
        p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=col_pillar,
            baseVisualShapeIndex=vis_pillar,
            basePosition=[px, py, half_h],
            physicsClientId=py_client,
        )


# ----- Exploration parameters -----
# Map = 6×6 box (world ±3 m); MAP_BOUND keeps goals inside
MAP_BOUND = MAP_SIZE / 2.0 - MAP_RES
GOAL_REACHED_DIST = 0.3
FRONTIER_MIN_NEIGHBORS = 2  # min unknown neighbors to count as frontier
PATH_WAYPOINT_DIST = 0.3   # consider waypoint reached when within this
BOUNDARY_BIAS = 1.5   # prefer goals with |x| or |y| >= this (explore outer area)
REPLAN_AVOID_THRESHOLD = 2.0   # replan path when avoidance force magnitude exceeds this

def get_frontier_unknown_cells():
    """Return unknown cells (0) that border free space (128) — candidates for exploration goals."""
    out = []
    for mx in range(1, MAP_DIM - 1):
        for my in range(1, MAP_DIM - 1):
            if occupancy_map[mx, my] != 0:
                continue  # not unknown
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    if occupancy_map[mx + dx, my + dy] == 128:
                        out.append((mx, my))
                        break
    return out


def sample_exploration_goal(drone_xy):
    """Pick next goal in unexplored (unknown) space: prefer frontier-unknown cells, else random unknown."""
    unexplored = get_frontier_unknown_cells()  # unknown cells that border free space
    drone_mx, drone_my = world_to_map(drone_xy[0], drone_xy[1])

    if unexplored:
        best = None
        best_score = -1.0
        for (mx, my) in unexplored:
            x, y = map_to_world(mx, my)
            if abs(x) > MAP_BOUND or abs(y) > MAP_BOUND:
                continue
            dist_from_drone = (mx - drone_mx) ** 2 + (my - drone_my) ** 2
            dist_from_center = max(abs(x), abs(y))
            boundary_bonus = 2.0 * dist_from_center if dist_from_center >= BOUNDARY_BIAS else 0.0
            score = dist_from_drone + boundary_bonus * 50
            if score > best_score:
                best_score = score
                best = np.array([x, y], dtype=float)
        if best is not None:
            return best

    # Fallback: pick random unknown cell
    candidates = []
    for _ in range(400):
        x = np.random.uniform(-MAP_BOUND, MAP_BOUND)
        y = np.random.uniform(-MAP_BOUND, MAP_BOUND)
        mx, my = world_to_map(x, y)
        if occupancy_map[mx, my] == 0:
            d2 = (x - drone_xy[0]) ** 2 + (y - drone_xy[1]) ** 2
            boundary_bonus = 30.0 if max(abs(x), abs(y)) >= BOUNDARY_BIAS else 0.0
            candidates.append((d2 + boundary_bonus, np.array([x, y], dtype=float)))
    if candidates:
        candidates.sort(key=lambda t: t[0], reverse=True)
        return candidates[0][1]
    return np.array([0.0, 0.0], dtype=float)


def plan_path(start_xy, goal_xy):
    """A* on occupancy grid. Passable = not obstacle (255). Returns list of world (x,y) waypoints."""
    smx, smy = world_to_map(start_xy[0], start_xy[1])
    gmx, gmy = world_to_map(goal_xy[0], goal_xy[1])

    def passable(mx, my):
        if mx < 0 or mx >= MAP_DIM or my < 0 or my >= MAP_DIM:
            return False
        return occupancy_map[mx, my] != 255

    if not passable(smx, smy) or not passable(gmx, gmy):
        return []

    # 8-neighbor moves
    moves = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    heap = [(0.0, 0, smx, smy)]
    cost_so_far = {(smx, smy): 0.0}
    parent = {}

    while heap:
        _, _, cx, cy = heapq.heappop(heap)
        if (cx, cy) == (gmx, gmy):
            break
        for dx, dy in moves:
            nx, ny = cx + dx, cy + dy
            if not passable(nx, ny):
                continue
            step = 1.414 if dx != 0 and dy != 0 else 1.0
            new_cost = cost_so_far.get((cx, cy), np.inf) + step
            if new_cost < cost_so_far.get((nx, ny), np.inf):
                cost_so_far[(nx, ny)] = new_cost
                parent[(nx, ny)] = (cx, cy)
                h = np.hypot(nx - gmx, ny - gmy)
                heapq.heappush(heap, (new_cost + h, 0, nx, ny))

    if (gmx, gmy) not in parent and (gmx, gmy) != (smx, smy):
        return []

    # Reconstruct path (start to goal in map indices)
    path_m = []
    cur = (gmx, gmy)
    while cur in parent:
        path_m.append(cur)
        cur = parent[cur]
    path_m.reverse()

    # Convert to world coords; subsample to avoid dense waypoints
    path_w = []
    step = max(1, len(path_m) // 25)
    for j in range(0, len(path_m), step):
        mx, my = path_m[j]
        path_w.append(np.array(map_to_world(mx, my), dtype=float))
    if path_m:
        mx, my = path_m[-1]
        path_w.append(np.array(map_to_world(mx, my), dtype=float))
    return path_w


def compute_avoidance_force(x, y, influence_radius=0.15):
    """Compute a simple repulsive force away from occupied cells (walls)."""
    mx, my = world_to_map(x, y)
    max_offset = int(influence_radius / MAP_RES)
    fx = 0.0
    fy = 0.0
    for ix in range(max(0, mx - max_offset), min(MAP_DIM, mx + max_offset + 1)):
        for iy in range(max(0, my - max_offset), min(MAP_DIM, my + max_offset + 1)):
            if occupancy_map[ix, iy] == 255:  # wall/obstacle
                cx, cy = map_to_world(ix, iy)
                dx = x - cx
                dy = y - cy
                d2 = dx * dx + dy * dy
                if d2 < 1e-4 or d2 > influence_radius * influence_radius:
                    continue
                inv = 1.0 / d2
                fx += dx * inv
                fy += dy * inv
    return np.array([fx, fy], dtype=float)


def simulate_lidar(py_client, drone_pos, num_rays=144, max_range=5.0):
    """Simulate a 2D lidar using PyBullet raycasts and update the occupancy map.

    Rays are cast in the horizontal plane. Free space along each ray is marked 128.
    Regular obstacles are marked 255, while fire cells are marked 200 (special but passable).

    The fire is detected geometrically from its known position (FIRE_POS) without relying on its
    collision shape, so the drone does not physically collide with it but lidar can still see it.
    """
    global FIRE_POS

    base = np.array([drone_pos[0], drone_pos[1], drone_pos[2]], dtype=float)
    ray_from = []
    ray_to = []
    ray_offset = 0.02  # start rays just outside drone body to avoid self-hit
    for k in range(num_rays):
        angle = 2.0 * np.pi * k / num_rays
        dx = np.cos(angle)
        dy = np.sin(angle)
        start = base + np.array([dx * ray_offset, dy * ray_offset, 0.0])
        end = base + np.array([dx * max_range, dy * max_range, 0.0])
        ray_from.append(start.tolist())
        ray_to.append(end.tolist())

    results = p.rayTestBatch(ray_from, ray_to, physicsClientId=py_client)
    fire_hit_pos = None
    FIRE_DETECT_RADIUS = 0.15

    for k, res in enumerate(results):
        # PyBullet hit information (used for regular obstacles)
        hit_fraction_pb = res[2]

        start = np.array(ray_from[k], dtype=float)
        end = np.array(ray_to[k], dtype=float)
        ray_vec = end - start
        ray_len = np.linalg.norm(ray_vec)
        if ray_len < 1e-6:
            continue

        # Determine if/where this ray intersects the fire disk in XY
        use_fraction = hit_fraction_pb
        is_fire_hit = False

        if FIRE_POS is not None:
            sx, sy = start[0], start[1]
            ex, ey = end[0], end[1]
            fx, fy = FIRE_POS[0], FIRE_POS[1]
            vx = ex - sx
            vy = ey - sy
            a = vx * vx + vy * vy
            if a > 1e-9:
                dx = sx - fx
                dy = sy - fy
                b = 2.0 * (dx * vx + dy * vy)
                c = dx * dx + dy * dy - FIRE_DETECT_RADIUS * FIRE_DETECT_RADIUS
                disc = b * b - 4.0 * a * c
                if disc >= 0.0:
                    sqrt_disc = np.sqrt(disc)
                    t1 = (-b - sqrt_disc) / (2.0 * a)
                    t2 = (-b + sqrt_disc) / (2.0 * a)
                    t_candidates = [t for t in (t1, t2) if 0.0 <= t <= 1.0]
                    if t_candidates:
                        t_fire = min(t_candidates)
                        if t_fire < use_fraction:
                            use_fraction = t_fire
                            is_fire_hit = True

        traveled = ray_len * use_fraction
        if traveled <= 0.0:
            continue

        # Step along ray at ~MAP_RES spacing; mark every cell as free (don't overwrite obstacles)
        num_steps = max(1, int(traveled / MAP_RES))
        for i in range(num_steps + 1):
            t = (i / num_steps) * use_fraction
            pt = start + ray_vec * t
            mx, my = world_to_map(pt[0], pt[1])
            if occupancy_map[mx, my] != 255:
                occupancy_map[mx, my] = 128  # free/observed

        # Mark hit point (fire or regular obstacle)
        if use_fraction < 1.0:
            hit_pos = start + ray_vec * use_fraction
            mx, my = world_to_map(hit_pos[0], hit_pos[1])
            if is_fire_hit:
                fire_hit_pos = hit_pos
                occupancy_map[mx, my] = 200  # special value for fire
            else:
                occupancy_map[mx, my] = 255  # regular obstacle

    return fire_hit_pos

def spawn_fire(py_client, drone_xy=None):
    """Spawn a random 'fire' cylinder that doesn't overlap any obstacle or the drone."""
    global FIRE_BODY_ID, FIRE_POS

    fire_height = 1.0
    fire_radius = 0.12
    half_h = fire_height / 2.0
    color = [1.0, 0.3, 0.0, 1.0]

    col = -1
    vis = p.createVisualShape(
        p.GEOM_CYLINDER,
        radius=fire_radius,
        length=fire_height,
        rgbaColor=color,
        physicsClientId=py_client,
    )

    pillar_clearance = fire_radius + 0.08 + 0.2
    wall_clearance = 0.4

    while True:
        fx = np.random.uniform(-MAP_BOUND * 0.8, MAP_BOUND * 0.8)
        fy = np.random.uniform(-MAP_BOUND * 0.8, MAP_BOUND * 0.8)

        if np.hypot(fx - INIT_XYZS[0, 0], fy - INIT_XYZS[0, 1]) < 0.5:
            continue
        if drone_xy is not None and np.hypot(fx - drone_xy[0], fy - drone_xy[1]) < 0.5:
            continue

        too_close = False
        for ppx, ppy in PILLAR_POSITIONS:
            if np.hypot(fx - ppx, fy - ppy) < pillar_clearance:
                too_close = True
                break
        if too_close:
            continue

        if abs(fx) > (3.0 - wall_clearance) or abs(fy) > (3.0 - wall_clearance):
            continue

        FIRE_POS = np.array([fx, fy, half_h], dtype=float)
        FIRE_BODY_ID = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=FIRE_POS.tolist(),
            physicsClientId=py_client,
        )
        break


def extinguish_fire(py_client):
    """Remove the current fire from the simulation and record its position."""
    global FIRE_BODY_ID, FIRE_POS

    if FIRE_POS is not None:
        EXTINGUISHED_FIRE_XY.append((FIRE_POS[0], FIRE_POS[1]))
    if FIRE_BODY_ID is not None:
        p.removeBody(FIRE_BODY_ID, physicsClientId=py_client)
        FIRE_BODY_ID = None
    FIRE_POS = None
    occupancy_map[occupancy_map == 200] = 128

# ----- Create environment -----
env = CtrlAviary(
    drone_model=DRONE,
    num_drones=NUM_DRONES,
    initial_xyzs=INIT_XYZS,
    initial_rpys=INIT_RPYS,
    physics=PHYSICS,
    neighbourhood_radius=5,
    pyb_freq=SIM_FREQ,
    ctrl_freq=CTRL_FREQ,
    gui=GUI,
    record=False,
    obstacles=OBSTACLES,
    user_debug_gui=False
)

PYB_CLIENT = env.getPyBulletClient()
create_maze(PYB_CLIENT)
spawn_fire(PYB_CLIENT)
print(f"Fire 1 spawned at ({FIRE_POS[0]:.2f}, {FIRE_POS[1]:.2f})")
ctrl = [DSLPIDControl(drone_model=DRONE) for _ in range(NUM_DRONES)]
# Softer position gains for stability (less overshoot, more damping)
for c in ctrl:
    c.P_COEFF_FOR = np.array([0.25, 0.25, 1.0])
    c.D_COEFF_FOR = np.array([0.35, 0.35, 0.6])
action = np.zeros((NUM_DRONES,4))

# ----- Autonomous exploration with SLAM -----
cmd_xy = np.array(INIT_XYZS[0, :2], dtype=float)
MAX_CMD_STEP = 1.0   # max (x,y) change per control step; larger = faster flight (~4.8 m/s at 48 Hz)
goal_xy = None
path = []   # list of (x,y) waypoints from plan_path
fire_goal_xy = None
fire_reached = False
fire_reached_steps = 0
fires_extinguished = 0
FIRE_SUCCESS_WAIT_STEPS = int(2.0 * CTRL_FREQ)  # hover ~2 seconds at fire to extinguish it

START = time.time()
try:
    for i in range(int(DURATION_SEC*CTRL_FREQ)):
        obs, _, _, _, _ = env.step(action)

        # Current position
        drone_pos = obs[0][:3]

        # ----- Lidar-based SLAM: cast rays, update occupancy, and detect fire -----
        fire_hit = simulate_lidar(PYB_CLIENT, drone_pos)
        if fire_hit is not None and fire_goal_xy is None:
            fire_goal_xy = fire_hit[:2]

        # ----- Choose / update exploration goal and path (fire has priority) -----
        if fire_goal_xy is not None and not fire_reached:
            desired_goal = fire_goal_xy
        else:
            desired_goal = goal_xy

        if desired_goal is None:
            desired_goal = sample_exploration_goal(drone_pos[:2])
            path = plan_path(drone_pos[:2], desired_goal)

        goal_xy = desired_goal

        dist_to_goal = np.linalg.norm(drone_pos[:2] - goal_xy)
        if dist_to_goal < GOAL_REACHED_DIST:
            # If this goal was the fire, mark it reached
            if fire_goal_xy is not None and not fire_reached:
                if np.linalg.norm(drone_pos[:2] - fire_goal_xy) < GOAL_REACHED_DIST:
                    fire_reached = True
            if fire_reached:
                # Loiter at fire once reached
                goal_xy = fire_goal_xy
                path = []
            else:
                goal_xy = sample_exploration_goal(drone_pos[:2])
                path = plan_path(drone_pos[:2], goal_xy)

        # Always account for newly mapped obstacles by replanning from current position
        avoid = compute_avoidance_force(drone_pos[0], drone_pos[1])
        if dist_to_goal > GOAL_REACHED_DIST:
            path = plan_path(drone_pos[:2], goal_xy)

        # Next waypoint: follow path if we have one, else aim at goal
        if path:
            target_xy = path[0].copy()
            if np.linalg.norm(drone_pos[:2] - path[0]) < PATH_WAYPOINT_DIST:
                path.pop(0)
                if path:
                    target_xy = path[0].copy()
                else:
                    target_xy = goal_xy.copy()
        else:
            target_xy = goal_xy.copy()
            if np.linalg.norm(drone_pos[:2] - goal_xy) > GOAL_REACHED_DIST:
                path = plan_path(drone_pos[:2], goal_xy)

        # ----- Move toward target (waypoint or goal) with wall avoidance -----
        to_target = target_xy - cmd_xy
        direction = to_target
        if np.linalg.norm(avoid) > 1e-4:
            direction = direction + 0.15 * avoid
        norm_dir = np.linalg.norm(direction)
        if norm_dir > 1e-6:
            step_vec = direction / norm_dir * MAX_CMD_STEP
        else:
            step_vec = np.zeros(2, dtype=float)
        if np.linalg.norm(step_vec) > np.linalg.norm(to_target):
            cmd_xy = target_xy.copy()
        else:
            cmd_xy += step_vec

        target_pos = np.array([cmd_xy[0], cmd_xy[1], FLIGHT_HEIGHT])
        action[0, :], _, _ = ctrl[0].computeControlFromState(
            control_timestep=env.CTRL_TIMESTEP,
            state=obs[0],
            target_pos=target_pos,
            target_rpy=INIT_RPYS[0, :]
        )

        if fire_reached:
            fire_reached_steps += 1
            if fire_reached_steps >= FIRE_SUCCESS_WAIT_STEPS:
                fires_extinguished += 1
                print(f"FIRE {fires_extinguished}/{TOTAL_FIRES} EXTINGUISHED!")
                extinguish_fire(PYB_CLIENT)

                if fires_extinguished >= TOTAL_FIRES:
                    print(f"SUCCESS: All {TOTAL_FIRES} fires extinguished!")
                    break

                spawn_fire(PYB_CLIENT, drone_pos[:2])
                print(f"New fire spawned at ({FIRE_POS[0]:.2f}, {FIRE_POS[1]:.2f}). Resuming exploration...")
                fire_goal_xy = None
                fire_reached = False
                fire_reached_steps = 0
                goal_xy = None
                path = []

        # ----- Render and sync -----
        env.render()
        if GUI:
            sync(i, START, env.CTRL_TIMESTEP)
finally:
    try:
        env.close()
    except Exception:
        pass

    print(f"\n{'='*40}")
    print(f"  RESULTS: {fires_extinguished}/{TOTAL_FIRES} fires extinguished")
    print(f"{'='*40}")

    vals, counts = np.unique(occupancy_map, return_counts=True)
    print("Occupancy map value distribution:")
    for v, c in zip(vals, counts):
        print(f"  {v:>3d} -> {c} cells")

    from matplotlib.colors import ListedColormap
    cmap_custom = ListedColormap([
        [0.0, 0.0, 0.0],   # 0   = unknown  -> black
        [0.5, 0.5, 0.5],   # 128 = free     -> gray
        [1.0, 0.6, 0.0],   # 200 = fire     -> orange
        [1.0, 1.0, 1.0],   # 255 = obstacle -> white
    ])
    display = np.zeros_like(occupancy_map, dtype=np.uint8)
    display[occupancy_map == 128] = 1
    display[occupancy_map == 200] = 2
    display[occupancy_map == 255] = 3

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(display.T, origin="lower", cmap=cmap_custom, vmin=0, vmax=3)

    for idx, (fx, fy) in enumerate(EXTINGUISHED_FIRE_XY):
        mx, my = world_to_map(fx, fy)
        ax.plot(mx, my, marker='*', color='red', markersize=14,
                markeredgecolor='yellow', markeredgewidth=0.8)
        ax.annotate(f"Fire {idx+1}", (mx, my), textcoords="offset points",
                    xytext=(6, 6), fontsize=8, color='red', fontweight='bold')

    ax.set_title(f"Occupancy Map  —  {fires_extinguished}/{TOTAL_FIRES} fires extinguished")
    plt.savefig("occupancy_map.png", dpi=150, bbox_inches="tight")
    plt.show()