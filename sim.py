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
DURATION_SEC = 60  # longer run to explore the whole map

# ----- Initial drone state -----
FLIGHT_HEIGHT = 0.5   # Altitude in m; avoid low hover (e.g. 0.1) for stability
INIT_XYZS = np.array([[0, 0, FLIGHT_HEIGHT]])
INIT_RPYS = np.array([[0, 0, 0]])

# ----- Occupancy map setup (6×6 m box, same as arena) -----
MAP_SIZE = 6
MAP_RES = 0.1
MAP_DIM = int(MAP_SIZE / MAP_RES)
occupancy_map = np.zeros((MAP_DIM, MAP_DIM), dtype=np.uint8)

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
    pillar_positions = [
        (-1.2, 1.2),
        (1.5, -0.8),
        (-0.8, -1.5),
        (1.8, 1.5),
        (-1.8, 0.0),
    ]
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
    for px, py in pillar_positions:
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


def simulate_lidar(py_client, drone_pos, num_rays=144, max_range=1.5):
    """Simulate a 2D lidar using PyBullet raycasts and update the occupancy map.

    Rays are cast in the horizontal plane. Free space along each ray is marked 128;
    the hit cell is marked 255 (obstacle). Ray start is offset from drone center so
    the ray does not hit the drone body.
    """
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

    for k, res in enumerate(results):
        hit_fraction = res[2]
        start = np.array(ray_from[k], dtype=float)
        end = np.array(ray_to[k], dtype=float)
        ray_vec = end - start
        ray_len = np.linalg.norm(ray_vec)
        if ray_len < 1e-6:
            continue

        traveled = ray_len * hit_fraction
        if traveled <= 0.0:
            continue

        # Step along ray at ~MAP_RES spacing; mark every cell as free (don't overwrite obstacles)
        num_steps = max(1, int(traveled / MAP_RES))
        for i in range(num_steps + 1):
            t = (i / num_steps) * hit_fraction
            pt = start + ray_vec * t
            mx, my = world_to_map(pt[0], pt[1])
            if occupancy_map[mx, my] != 255:
                occupancy_map[mx, my] = 128  # free/observed

        # Mark obstacle at hit point
        if hit_fraction < 1.0:
            hit_pos = start + ray_vec * hit_fraction
            mx, my = world_to_map(hit_pos[0], hit_pos[1])
            occupancy_map[mx, my] = 255

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
ctrl = [DSLPIDControl(drone_model=DRONE) for _ in range(NUM_DRONES)]
# Softer position gains for stability (less overshoot, more damping)
for c in ctrl:
    c.P_COEFF_FOR = np.array([0.25, 0.25, 1.0])
    c.D_COEFF_FOR = np.array([0.35, 0.35, 0.6])
action = np.zeros((NUM_DRONES,4))

# ----- Autonomous exploration with SLAM -----
cmd_xy = np.array(INIT_XYZS[0, :2], dtype=float)
MAX_CMD_STEP = 5.0   # max (x,y) change per control step; larger = faster flight (~4.8 m/s at 48 Hz)
goal_xy = None
path = []   # list of (x,y) waypoints from plan_path

START = time.time()
try:
    for i in range(int(DURATION_SEC*CTRL_FREQ)):
        obs, _, _, _, _ = env.step(action)

        # Current position
        drone_pos = obs[0][:3]

        # ----- Choose / update exploration goal and path (only when no goal or goal reached) -----
        if goal_xy is None:
            goal_xy = sample_exploration_goal(drone_pos[:2])
            path = plan_path(drone_pos[:2], goal_xy)

        dist_to_goal = np.linalg.norm(drone_pos[:2] - goal_xy)
        if dist_to_goal < GOAL_REACHED_DIST:
            goal_xy = sample_exploration_goal(drone_pos[:2])
            path = plan_path(drone_pos[:2], goal_xy)

        # If repelled from wall, replan path from current position to goal
        avoid = compute_avoidance_force(drone_pos[0], drone_pos[1])
        if np.linalg.norm(avoid) > REPLAN_AVOID_THRESHOLD and dist_to_goal > GOAL_REACHED_DIST:
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

        # ----- Lidar-based SLAM: cast rays and update occupancy -----
        simulate_lidar(PYB_CLIENT, drone_pos)

        # ----- Render and sync -----
        env.render()
        if GUI:
            sync(i, START, env.CTRL_TIMESTEP)
finally:
    try:
        env.close()
    except Exception:
        pass
    #np.save("occupancy_map.npy", occupancy_map)
    plt.figure(figsize=(6, 6))
    plt.imshow(occupancy_map.T, origin="lower", cmap="gray")
    plt.title("Occupancy Map")
    plt.savefig("occupancy_map.png", dpi=150, bbox_inches="tight")
    plt.show()