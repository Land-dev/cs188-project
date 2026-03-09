"""
Path planning on a 2D occupancy grid: coordinate conversion and A*.
Occupancy: 0 = unknown, 128 = free, 200 = fire, 255 = obstacle.
"""
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


def _inflate_obstacles(occupancy_map, map_res, safe_margin, goal_cell=None):
    """
    Create a cost map where cells near obstacles get a high penalty.
    Uses scipy binary_dilation for speed instead of nested Python loops.
    Returns a 2D float array of additional cost per cell (0.0 = safe, large = near obstacle).
    """
    map_dim = occupancy_map.shape[0]
    inflation_cells = int(np.ceil(safe_margin / map_res))
    if inflation_cells <= 0:
        return np.zeros((map_dim, map_dim), dtype=np.float32)

    # Build a circular structuring element
    diameter = 2 * inflation_cells + 1
    struct = np.zeros((diameter, diameter), dtype=bool)
    for dx in range(-inflation_cells, inflation_cells + 1):
        for dy in range(-inflation_cells, inflation_cells + 1):
            if dx * dx + dy * dy <= inflation_cells * inflation_cells:
                struct[dx + inflation_cells, dy + inflation_cells] = True

    # Build obstacle mask (255 = wall, 200 = fire treated as obstacle for inflation)
    obstacle_mask = (occupancy_map == 255) | (occupancy_map == 200)

    # Exclude the goal cell from inflation if it's a fire
    if goal_cell is not None:
        gx, gy = goal_cell
        if 0 <= gx < map_dim and 0 <= gy < map_dim:
            if occupancy_map[gx, gy] == 200:
                obstacle_mask[gx, gy] = False

    # Dilate
    inflated = binary_dilation(obstacle_mask, structure=struct)

    # Cost: 0 where safe, large penalty where inside inflated zone but not a real obstacle
    cost_map = np.zeros((map_dim, map_dim), dtype=np.float32)
    cost_map[inflated & ~(occupancy_map == 255)] = 50.0  # penalty, not impassable
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

    # Precompute inflation cost map once per call
    if safe_margin > 0.0:
        cost_map = _inflate_obstacles(occupancy_map, map_res, safe_margin, goal_cell=(gmx, gmy))
    else:
        cost_map = None

    def passable(mx, my):
        if mx < 0 or mx >= map_dim or my < 0 or my >= map_dim:
            return False
        val = occupancy_map[mx, my]
        if val == 255:
            return False
        if val == 200:  # fire: only the goal cell is passable
            return (mx, my) == (gmx, gmy)
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

            # Add soft margin penalty
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

    # Subsample waypoints but keep every ~5 cells instead of 1/25th to avoid skipping around obstacles
    path_w = []
    step = max(1, len(path_m) // 40)
    for j in range(0, len(path_m), step):
        mx, my = path_m[j]
        path_w.append(np.array(map_to_world(mx, my, map_size, map_res), dtype=float))
    if path_m:
        mx, my = path_m[-1]
        path_w.append(np.array(map_to_world(mx, my, map_size, map_res), dtype=float))
    return path_w
