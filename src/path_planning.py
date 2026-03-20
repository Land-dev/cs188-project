"""A* path planning on a 2D occupancy grid. Values: 0=unknown, 128=free, 200=fire, 255=obstacle."""
import heapq
import numpy as np
from scipy.ndimage import binary_dilation, generate_binary_structure


def world_to_map(x, y, map_size, map_res):
    """Convert world (x, y) to grid indices (mx, my)."""
    map_dim = int(map_size / map_res)
    mx = int((x + map_size / 2) / map_res)
    my = int((y + map_size / 2) / map_res)
    return np.clip(mx, 0, map_dim - 1), np.clip(my, 0, map_dim - 1)


def map_to_world(mx, my, map_size, map_res):
    """Convert grid indices (mx, my) to world (x, y)."""
    x = (mx + 0.5) * map_res - map_size / 2.0
    y = (my + 0.5) * map_res - map_size / 2.0
    return x, y


def path_hits_obstacle(occupancy_map, map_size, map_res, start_xy, path, goal_xy):
    """
    Check if the path from start through waypoints to goal crosses any obstacle (255).
    Samples points along each segment at map resolution.
    """
    waypoints = [np.array(start_xy, dtype=float)] + [np.array(w, dtype=float) for w in path] + [np.array(goal_xy, dtype=float)]
    for i in range(len(waypoints) - 1):
        a, b = waypoints[i], waypoints[i + 1]
        seg_len = np.linalg.norm(b - a)
        if seg_len < 1e-6:
            continue
        num_steps = max(2, int(seg_len / map_res) + 1)
        for j in range(num_steps + 1):
            t = j / num_steps
            pt = a + t * (b - a)
            mx, my = world_to_map(pt[0], pt[1], map_size, map_res)
            if occupancy_map[mx, my] == 255:
                return True
    return False


def _inflate_obstacles(occupancy_map, map_res, safe_margin, goal_cell=None):
    """Return a cost map with penalty=50 for cells within safe_margin of obstacles."""
    map_dim = occupancy_map.shape[0]
    inflation_cells = int(np.ceil(safe_margin / map_res))
    if inflation_cells <= 0:
        return np.zeros((map_dim, map_dim), dtype=np.float32)

    diameter = 2 * inflation_cells + 1
    struct = np.zeros((diameter, diameter), dtype=bool)
    for dx in range(-inflation_cells, inflation_cells + 1):
        for dy in range(-inflation_cells, inflation_cells + 1):
            if dx * dx + dy * dy <= inflation_cells * inflation_cells:
                struct[dx + inflation_cells, dy + inflation_cells] = True

    # Only inflate physical obstacles (255), not fire cells (200) — inflating fire
    # creates a cost barrier that blocks the drone from reaching it.
    obstacle_mask = (occupancy_map == 255)
    inflated = binary_dilation(obstacle_mask, structure=struct)

    cost_map = np.zeros((map_dim, map_dim), dtype=np.float32)
    cost_map[inflated & ~obstacle_mask] = 50.0
    return cost_map


def plan_path(occupancy_map, map_size, map_res, start_xy, goal_xy, safe_margin=0.0):
    """
    A* on occupancy grid. Obstacles = 255; fire (200) is obstacle except goal cell.
    If safe_margin > 0.0, cells near obstacles get a cost penalty (soft margin).
    Returns list of world (x, y) waypoints, or empty list if no path.
    """
    map_dim = int(map_size / map_res)
    smx, smy = world_to_map(start_xy[0], start_xy[1], map_size, map_res)
    gmx, gmy = world_to_map(goal_xy[0], goal_xy[1], map_size, map_res)

    if safe_margin > 0.0:
        cost_map = _inflate_obstacles(occupancy_map, map_res, safe_margin, goal_cell=(gmx, gmy))
    else:
        cost_map = None

    goal_is_fire = (0 <= gmx < map_dim and 0 <= gmy < map_dim and
                    occupancy_map[gmx, gmy] == 200)

    def passable(mx, my):
        if mx < 0 or mx >= map_dim or my < 0 or my >= map_dim:
            return False
        val = occupancy_map[mx, my]
        if val == 255:
            return False
        if val == 200:
            return goal_is_fire  # fire cells only passable when navigating to fire
        return True

    if not passable(smx, smy) or not passable(gmx, gmy):
        return []

    moves = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
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
            if cost_map is not None:
                step += cost_map[nx, ny]

            new_cost = cost_so_far.get((cx, cy), np.inf) + step
            if new_cost < cost_so_far.get((nx, ny), np.inf):
                cost_so_far[(nx, ny)] = new_cost
                parent[(nx, ny)] = (cx, cy)
                h = np.hypot(nx - gmx, ny - gmy)
                heapq.heappush(heap, (new_cost + h, 0, nx, ny))

    if (gmx, gmy) not in parent and (gmx, gmy) != (smx, smy):
        return []

    path_m = []
    cur = (gmx, gmy)
    while cur in parent:
        path_m.append(cur)
        cur = parent[cur]
    path_m.reverse()

    path_w = []
    step = max(1, len(path_m) // 40)
    for j in range(0, len(path_m), step):
        mx, my = path_m[j]
        path_w.append(np.array(map_to_world(mx, my, map_size, map_res), dtype=float))
    if path_m:
        mx, my = path_m[-1]
        path_w.append(np.array(map_to_world(mx, my, map_size, map_res), dtype=float))
    return path_w
