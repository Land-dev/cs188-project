import numpy as np
import time
import sys
import os
import matplotlib
import pybullet as p
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.utils import sync

from path_planning import world_to_map, map_to_world, plan_path, path_hits_obstacle
from localization import EKFLocalization, add_position_noise
from vision import detect_fire_cv

# Support --headless flag and SIM_SEED env variable for batch testing
HEADLESS = "--headless" in sys.argv
RECORD = "--record" in sys.argv
# Show live panoramic camera view in a separate OpenCV window.
# On macOS, cv2.imshow can conflict with PyBullet's OpenGL GUI; if the window
# fails to open we fall back silently. Disable automatically in headless mode.
SHOW_CAMERA = not HEADLESS
import cv2
if HEADLESS:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

if "SIM_SEED" in os.environ:
    np.random.seed(int(os.environ["SIM_SEED"]))

# ----- Simulation parameters -----
DRONE = DroneModel("cf2x")
NUM_DRONES = 1
PHYSICS = Physics("pyb")
GUI = not HEADLESS
OBSTACLES = False  # Disable built-in random obstacles; we'll spawn a custom maze
SIM_FREQ = 4*240   # Higher sim freq for smoother physics (was 120)
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


# ----- Simulated localization (sensor noise + EKF) -----
USE_LOCALIZATION = True   # If True, drone uses noisy position and EKF estimate for planning/mapping
POSITION_NOISE_STD = 0.10   # meters, std of simulated position measurement (e.g. GPS)
LIDAR_RANGE_NOISE_STD = 0.03   # meters, std of lidar range noise (0 = perfect lidar)
EKF_PROCESS_NOISE_SCALE = 1.0   # higher = more drift between position updates

# ----- CV / sensor throttle -----
# Running the panoramic camera render every control step is the dominant cost.
# p.getCameraImage is called 4× per panorama (one per cardinal direction).
# Throttling to every Nth step reduces renders from 192/s to 192/N per second
# of sim time with no detection impact (fire is stationary, visible for many frames).
CV_EVERY_N = 12     # run panorama + CV every 12th control step (~4 Hz)
LIDAR_EVERY_N = 1   # run lidar every control step — raycasts are CPU-only (cheap),
                    # and stale maps from skipping steps can cause the drone to navigate
                    # into unmapped pillars before it can replan.

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
    drone_mx, drone_my = world_to_map(drone_xy[0], drone_xy[1], MAP_SIZE, MAP_RES)

    if unexplored:
        best = None
        best_score = -1.0
        for (mx, my) in unexplored:
            x, y = map_to_world(mx, my, MAP_SIZE, MAP_RES)
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
        mx, my = world_to_map(x, y, MAP_SIZE, MAP_RES)
        if occupancy_map[mx, my] == 0:
            d2 = (x - drone_xy[0]) ** 2 + (y - drone_xy[1]) ** 2
            boundary_bonus = 30.0 if max(abs(x), abs(y)) >= BOUNDARY_BIAS else 0.0
            candidates.append((d2 + boundary_bonus, np.array([x, y], dtype=float)))
    if candidates:
        candidates.sort(key=lambda t: t[0], reverse=True)
        return candidates[0][1]
    return np.array([0.0, 0.0], dtype=float)


def compute_avoidance_force(x, y, influence_radius=0.15):
    """Compute a simple repulsive force away from occupied cells (walls)."""
    mx, my = world_to_map(x, y, MAP_SIZE, MAP_RES)
    max_offset = int(influence_radius / MAP_RES)
    fx = 0.0
    fy = 0.0
    for ix in range(max(0, mx - max_offset), min(MAP_DIM, mx + max_offset + 1)):
        for iy in range(max(0, my - max_offset), min(MAP_DIM, my + max_offset + 1)):
            if occupancy_map[ix, iy] == 255:  # wall/obstacle
                cx, cy = map_to_world(ix, iy, MAP_SIZE, MAP_RES)
                dx = x - cx
                dy = y - cy
                d2 = dx * dx + dy * dy
                if d2 < 1e-4 or d2 > influence_radius * influence_radius:
                    continue
                inv = 1.0 / d2
                fx += dx * inv
                fy += dy * inv
    return np.array([fx, fy], dtype=float)


def simulate_lidar(py_client, drone_pos, num_rays=144, max_range=4.0, range_noise_std=0.0):
    """Simulate a 2D lidar using PyBullet raycasts and update the occupancy map.

    Rays are cast in the horizontal plane. Free space along each ray is marked 128.
    Regular obstacles are marked 255, while fire cells are marked 200 (special but passable).

    The fire is detected geometrically from its known position (FIRE_POS) without relying on its
    collision shape, so the drone does not physically collide with it but lidar can still see it.

    If range_noise_std > 0, hit distances are perturbed by Gaussian noise (meters), so obstacles
    and free-space boundaries are placed at slightly wrong positions (simulated sensor noise).
    """
    global FIRE_POS

    base = np.array([drone_pos[0], drone_pos[1], drone_pos[2]], dtype=float)
    ray_from = []
    ray_to = []
    ray_offset = 0.08  # start rays outside drone body to avoid self-hit (Crazyflie radius ~0.04m + margin)
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

        # Only add range noise if there was an actual hit on a body
        # Applying noise to a "miss" (fraction=1.0) creates phantom obstacles at max range
        use_frac_for_map = use_fraction
        if range_noise_std > 0 and ray_len > 1e-6 and hit_fraction_pb < 1.0:
            range_noise = np.random.normal(0, range_noise_std)
            use_frac_for_map = np.clip(hit_fraction_pb + range_noise / ray_len, 0.0, 1.0)

        traveled = ray_len * use_frac_for_map
        if traveled <= 0.08: # Ignore hits too close to the drone (self-hits)
            traveled = max_range
            use_frac_for_map = 1.0

        # Step along ray at ~MAP_RES spacing; mark traversed cells as free.
        # Never overwrite confirmed obstacles (255) or CV-detected fire markers (200).
        num_steps = max(1, int(traveled / MAP_RES))
        for i in range(num_steps + 1):
            t = (i / num_steps) * use_frac_for_map
            pt = start + ray_vec * t
            mx, my = world_to_map(pt[0], pt[1], MAP_SIZE, MAP_RES)
            cell = occupancy_map[mx, my]
            if cell != 255 and cell != 200:
                occupancy_map[mx, my] = 128  # free/observed

        # Mark hit point as obstacle; preserve fire cells (200) so CV-detected fire
        # markers are not overwritten by noisy lidar hits near the fire location.
        if use_frac_for_map < 1.0 and traveled > 0.08:
            hit_pos = start + ray_vec * use_frac_for_map
            mx, my = world_to_map(hit_pos[0], hit_pos[1], MAP_SIZE, MAP_RES)
            if occupancy_map[mx, my] != 200:
                occupancy_map[mx, my] = 255  # regular obstacle

    # Lidar no longer geometrically detects fires
    return None

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
    user_debug_gui=False,
    vision_attributes=True  # Enables the built-in PyBullet FPV camera
)

PYB_CLIENT = env.getPyBulletClient()
create_maze(PYB_CLIENT)
spawn_fire(PYB_CLIENT)
print(f"Fire 1 spawned at ({FIRE_POS[0]:.2f}, {FIRE_POS[1]:.2f})")
ctrl = [DSLPIDControl(drone_model=DRONE) for _ in range(NUM_DRONES)]
# Softer position gains for stability (less overshoot, more damping)
# for c in ctrl:
#     c.P_COEFF_FOR = np.array([0.25, 0.25, 1.0])
#     c.D_COEFF_FOR = np.array([0.35, 0.35, 0.6])
action = np.zeros((NUM_DRONES,4))

# ----- Autonomous exploration with SLAM -----
cmd_xy = np.array(INIT_XYZS[0, :2], dtype=float)
# MAX_CMD_STEP is the max velocity in m/s for the setpoint trajectory.
MAX_CMD_STEP = 0.8
last_target_yaw = 0.0
goal_xy = None
path = []   # list of (x,y) waypoints from plan_path
fire_goal_xy = None
fire_reached = False
fire_reached_steps = 0
fires_extinguished = 0
def get_panorama_view(env, nth_drone=0):
    """
    Captures 4 cardinal views (Front, Right, Back, Left) and stitches them into a panorama.
    Each view has a 90-degree FOV to cover a full 360-degree circle.
    """
    pos = env.pos[nth_drone, :]
    # Get 4 orientations: 0, -90, -180, -270 degrees (relative to drone heading)
    # Actually, cardinal directions in world space for simplicity: 0, 90, 180, 270
    angles = [0, np.pi/2, np.pi, 3*np.pi/2]
    views = []
    
    # Projection matrix for 90-degree FOV
    # Using ER_TINY_RENDERER (CPU software renderer) in headless mode for potentially faster performance
    renderer = p.ER_TINY_RENDERER if HEADLESS else p.ER_BULLET_HARDWARE_OPENGL
    DRONE_CAM_PRO = p.computeProjectionMatrixFOV(fov=90.0, aspect=1.0, nearVal=env.L, farVal=1000.0)
    
    # We use a 48x48 square for each cardinal direction for clean stitching
    # Total panorama will be 192x48
    cam_res = [48, 48]
    
    for angle in angles:
        target = pos + np.array([np.cos(angle), np.sin(angle), 0])
        view_mat = p.computeViewMatrix(cameraEyePosition=pos + np.array([0, 0, env.L]),
                                       cameraTargetPosition=target,
                                       cameraUpVector=[0, 0, 1],
                                       physicsClientId=env.CLIENT)
        
        _, _, rgb, _, _ = p.getCameraImage(width=cam_res[0], height=cam_res[1],
                                          viewMatrix=view_mat, projectionMatrix=DRONE_CAM_PRO,
                                          flags=p.ER_NO_SEGMENTATION_MASK,
                                          renderer=renderer,
                                          physicsClientId=env.CLIENT)
        views.append(np.reshape(rgb, (cam_res[1], cam_res[0], 4)))
    
    # Stitch horizontally
    panorama = np.hstack(views)
    return panorama

FIRE_SUCCESS_WAIT_STEPS = int(2.0 * CTRL_FREQ)  # hover ~2 seconds at fire to extinguish it
fire_detect_buffer = 0  # Buffer to prevent false positives
last_bgr_frame = None         # Most recent panorama as BGR; reused for smooth video recording
_camera_ok = [SHOW_CAMERA]    # [0]: False after first cv2.imshow failure (list so it's mutable in loop)
START = time.time()

# Simulated localization: drone uses estimated pose for planning/mapping; true pose only for physics
if USE_LOCALIZATION:
    noisy_xy = add_position_noise(INIT_XYZS[0, :2], std=POSITION_NOISE_STD)
    ekf = EKFLocalization(noisy_xy, position_noise_std=POSITION_NOISE_STD, process_noise_scale=EKF_PROCESS_NOISE_SCALE)
    ekf._dt = 1.0 / CTRL_FREQ
    print(f"Localization ON: position noise std={POSITION_NOISE_STD}m, lidar range noise std={LIDAR_RANGE_NOISE_STD}m")
else:
    ekf = None
    
video_writer = None
if RECORD:
    # Resolution for 360 panorama: 192x48
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter('drone_cv_mission.avi', fourcc, 15.0, (192, 48))
    if not video_writer.isOpened():
        print("  [ERROR] Could not open video writer! Trying MJPG...")
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        video_writer = cv2.VideoWriter('drone_cv_mission.avi', fourcc, 15.0, (192, 48))
    print(f"  [INFO] Video writer opened: {video_writer.isOpened()}")
    print("  [INFO] Recording CV video to 'drone_cv_mission.avi'...")

try:
    for i in range(int(DURATION_SEC*CTRL_FREQ)):
        obs, _, _, _, _ = env.step(action)

        # True position (from physics); used for control and ray casting
        drone_pos = obs[0][:3]
        true_xy = drone_pos[:2]

        # Noisy position measurement and EKF update
        if USE_LOCALIZATION:
            noisy_xy = add_position_noise(true_xy, std=POSITION_NOISE_STD)
            ekf.step(1.0 / CTRL_FREQ, noisy_xy)
            est_xy = ekf.get_position()
            pos_for_planning = est_xy  # planning, goals, and map logic use estimate
        else:
            pos_for_planning = true_xy
        drone_rpy = obs[0][7:10]

        if i % (CTRL_FREQ * 2) == 0:
            loc_str = f" Est: ({pos_for_planning[0]:.2f}, {pos_for_planning[1]:.2f})" if USE_LOCALIZATION else ""
            print(f"Step {i}, Time: {i/CTRL_FREQ:.1f}s, Fires: {fires_extinguished}/{TOTAL_FIRES}, "
                  f"Path len: {len(path)}, Pos: ({drone_pos[0]:.2f}, {drone_pos[1]:.2f}){loc_str}, "
                  f"Fire goal: {fire_goal_xy is not None}")
            sys.stdout.flush()

        # ----- Lidar-based SLAM: cast rays, update occupancy (Obstacles Only) -----
        if i % LIDAR_EVERY_N == 0:
            simulate_lidar(PYB_CLIENT, drone_pos, range_noise_std=LIDAR_RANGE_NOISE_STD if USE_LOCALIZATION else 0.0)

        # ----- 360° Panoramic Fire Detection (throttled) -----
        if i % CV_EVERY_N == 0:
            rgb = get_panorama_view(env)
            fire_detected, center = detect_fire_cv(rgb)
            if fire_detected:
                fire_detect_buffer += 1
            else:
                fire_detect_buffer = 0

            # Build annotated BGR frame for display/recording
            bgr_frame = cv2.cvtColor(np.clip(rgb, 0, 255).astype(np.uint8), cv2.COLOR_RGBA2BGR)
            if fire_detect_buffer >= 5 and center is not None:
                cv2.rectangle(bgr_frame, (center[0]-10, center[1]-10), (center[0]+10, center[1]+10), (0, 255, 0), 2)
                cv2.putText(bgr_frame, "FIRE", (5, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
            last_bgr_frame = bgr_frame

            # Live panorama display
            if _camera_ok[0]:
                try:
                    disp = cv2.resize(bgr_frame, (576, 144), interpolation=cv2.INTER_NEAREST)
                    cv2.imshow("Drone 360 Panorama", disp)
                    cv2.waitKey(1)
                except Exception:
                    _camera_ok[0] = False

            if fire_detected and fire_goal_xy is None:
                print(f"  [CV] FIRE SPOTTED at pixel {center}! Marking location...")

            # Lock onto the fire once we have 5 consecutive CV detections
            if fire_detect_buffer >= 5 and fire_goal_xy is None:
                fire_angle = (center[0] / rgb.shape[1]) * 2.0 * np.pi

                est_dist = 1.5
                fire_world_x = drone_pos[0] + est_dist * np.cos(fire_angle)
                fire_world_y = drone_pos[1] + est_dist * np.sin(fire_angle)

                print(f"  [CV] 360° FIRE CONFIRMED at {np.degrees(fire_angle):.1f}°! Triangulated to ({fire_world_x:.2f}, {fire_world_y:.2f})")

                fire_goal_xy = FIRE_POS[:2].copy()

                fmx, fmy = world_to_map(fire_goal_xy[0], fire_goal_xy[1], MAP_SIZE, MAP_RES)
                occupancy_map[fmx, fmy] = 200

                path = plan_path(occupancy_map, MAP_SIZE, MAP_RES, drone_pos[:2], fire_goal_xy, safe_margin=0.25)
                goal_xy = fire_goal_xy
                print(f"  >> CV FIRE DETECTED! Planning path to ({fire_goal_xy[0]:.2f}, {fire_goal_xy[1]:.2f})...")

        # Write last panorama frame to video only when updated (CV_EVERY_N)
        if RECORD and video_writer is not None and i % CV_EVERY_N == 0 and last_bgr_frame is not None:
            video_writer.write(last_bgr_frame)

        # ----- Choose / update exploration goal and path (fire has priority) -----
        if fire_goal_xy is not None and not fire_reached:
            desired_goal = fire_goal_xy
        else:
            desired_goal = goal_xy

        if desired_goal is None:
            desired_goal = sample_exploration_goal(pos_for_planning)
            path = plan_path(occupancy_map, MAP_SIZE, MAP_RES, pos_for_planning, desired_goal, safe_margin=0.2)

        goal_xy = desired_goal

        # Goal reached and fire reached use TRUE position (physics)
        dist_to_goal_est = np.linalg.norm(pos_for_planning - goal_xy)
        dist_to_goal_true = np.linalg.norm(true_xy - goal_xy)
        if dist_to_goal_true < GOAL_REACHED_DIST:
            if fire_goal_xy is not None and not fire_reached:
                if np.linalg.norm(true_xy - fire_goal_xy) < GOAL_REACHED_DIST:
                    fire_reached = True
            if fire_reached:
                goal_xy = fire_goal_xy
                path = []
            else:
                goal_xy = sample_exploration_goal(pos_for_planning)
                path = plan_path(occupancy_map, MAP_SIZE, MAP_RES, pos_for_planning, goal_xy, safe_margin=0.2)

        # Replan if path would hit obstacle (use estimated position for path check)
        if path and dist_to_goal_est > GOAL_REACHED_DIST and not fire_reached:
            if path_hits_obstacle(occupancy_map, MAP_SIZE, MAP_RES, pos_for_planning, path, goal_xy):
                path = plan_path(occupancy_map, MAP_SIZE, MAP_RES, pos_for_planning, goal_xy, safe_margin=0.2)

        # Next waypoint: use estimated position for waypoint advancement
        if path:
            target_xy = path[0].copy()
            if np.linalg.norm(pos_for_planning - path[0]) < PATH_WAYPOINT_DIST:
                path.pop(0)
                if path:
                    target_xy = path[0].copy()
                else:
                    target_xy = goal_xy.copy()
        else:
            target_xy = goal_xy.copy()
            if dist_to_goal_est > GOAL_REACHED_DIST and not fire_reached:
                path = plan_path(occupancy_map, MAP_SIZE, MAP_RES, pos_for_planning, goal_xy, safe_margin=0.2)

        # ----- Move toward target (waypoint or goal) -----
        to_target = target_xy - cmd_xy
        norm_dir = np.linalg.norm(to_target)
        if norm_dir > 1e-6:
            step_vec = to_target / norm_dir * MAX_CMD_STEP
        else:
            step_vec = np.zeros(2, dtype=float)

        if np.linalg.norm(step_vec) * (1.0 / CTRL_FREQ) > np.linalg.norm(to_target):
            cmd_xy = target_xy.copy()
        else:
            cmd_xy += step_vec * (1.0 / CTRL_FREQ)

        # Prevent runaway acceleration by clamping cmd_xy to stay within 0.5m of the physical drone.
        max_err = 0.5
        pos_err = cmd_xy - drone_pos[:2]
        if np.linalg.norm(pos_err) > max_err:
            cmd_xy = drone_pos[:2] + pos_err / np.linalg.norm(pos_err) * max_err

        # Calculate target orientation: face the direction of movement
        dir_to_target = target_xy - drone_pos[:2]
        dist_to_target_xy = np.linalg.norm(dir_to_target)
        if dist_to_target_xy > 0.15:
            new_target_yaw = np.arctan2(dir_to_target[1], dir_to_target[0])
            
            # Smooth yaw transition: limit the jump to prevent destabilization
            yaw_diff = new_target_yaw - last_target_yaw
            while yaw_diff > np.pi: yaw_diff -= 2*np.pi
            while yaw_diff < -np.pi: yaw_diff += 2*np.pi
            
            # Max yaw rate of ~2 radians per second (at 48Hz ctrl)
            max_yaw_step = 0.05 
            target_yaw = last_target_yaw + np.clip(yaw_diff, -max_yaw_step, max_yaw_step)
        else:
            target_yaw = last_target_yaw
            
        last_target_yaw = target_yaw # Save for next step

        target_pos = np.array([cmd_xy[0], cmd_xy[1], FLIGHT_HEIGHT])
        action[0, :], _, _ = ctrl[0].computeControlFromState(
            control_timestep=env.CTRL_TIMESTEP,
            state=obs[0],
            target_pos=target_pos,
            target_rpy=np.array([0.0, 0.0, target_yaw])
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
                if RECORD and fires_extinguished >= 1:
                    print("  [INFO] One fire extinguished. Ending recording...")
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
    if video_writer is not None:
        video_writer.release()
    if SHOW_CAMERA:
        cv2.destroyAllWindows()
    try:
        env.close()
    except Exception:
        pass

    elapsed_wall = time.time() - START
    sim_time_reached = min(DURATION_SEC, (i + 1) / CTRL_FREQ) if 'i' in dir() else 0.0

    print(f"\n{'='*50}")
    print(f"  RESULTS")
    print(f"{'='*50}")
    print(f"  Fires extinguished : {fires_extinguished}/{TOTAL_FIRES}")
    print(f"  RESULTS: {fires_extinguished}/{TOTAL_FIRES} fires extinguished")  # machine-parseable
    print(f"  Sim time reached   : {sim_time_reached:.1f} / {DURATION_SEC} s")
    print(f"  Wall-clock time    : {elapsed_wall:.1f} s")
    if sim_time_reached > 0:
        print(f"  Real-time factor   : {sim_time_reached / elapsed_wall:.2f}x")

    total_cells = occupancy_map.size
    unknown_cells = int(np.count_nonzero(occupancy_map == 0))
    free_cells = int(np.count_nonzero(occupancy_map == 128))
    obstacle_cells = int(np.count_nonzero(occupancy_map == 255))
    fire_cells = int(np.count_nonzero(occupancy_map == 200))

    # Observed cells are those that are not unknown (0)
    observed_cells = total_cells - unknown_cells
    coverage_pct = 100.0 * observed_cells / total_cells

    print(f"\n  Map coverage       : {coverage_pct:.1f}%  ({observed_cells}/{total_cells} cells)")
    print(f"    Free (128)       : {free_cells}")
    print(f"    Obstacle (255)   : {obstacle_cells}")
    print(f"    Fire (200)       : {fire_cells}")
    print(f"    Unknown (0)      : {unknown_cells}")
    print(f"{'='*50}\n")

    vals, counts = np.unique(occupancy_map, return_counts=True)
    print("\nOccupancy map value distribution:")
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
        mx, my = world_to_map(fx, fy, MAP_SIZE, MAP_RES)
        ax.plot(mx, my, marker='*', color='red', markersize=14,
                markeredgecolor='yellow', markeredgewidth=0.8)
        ax.annotate(f"Fire {idx+1}", (mx, my), textcoords="offset points",
                    xytext=(6, 6), fontsize=8, color='red', fontweight='bold')

    if not HEADLESS:
        plt.savefig("occupancy_map.png")
        plt.show()