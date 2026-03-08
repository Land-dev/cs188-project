"""
Path planning on a 2D occupancy grid: coordinate conversion and A*.
Occupancy: 0 = unknown, 128 = free, 200 = fire, 255 = obstacle.
"""
import heapq
import numpy as np


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


def plan_path(occupancy_map, map_size, map_res, start_xy, goal_xy):
    """
    A* on occupancy grid. Obstacles = 255; fire (200) is obstacle except goal cell.
    Returns list of world (x, y) waypoints, or empty list if no path.
    """
    map_dim = int(map_size / map_res)
    smx, smy = world_to_map(start_xy[0], start_xy[1], map_size, map_res)
    gmx, gmy = world_to_map(goal_xy[0], goal_xy[1], map_size, map_res)

    def passable(mx, my):
        if mx < 0 or mx >= map_dim or my < 0 or my >= map_dim:
            return False
        val = occupancy_map[mx, my]
        if val == 255:
            return False
        if val == 200:  # fire: only the goal cell is passable so we path around and stop at center
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
    step = max(1, len(path_m) // 25)
    for j in range(0, len(path_m), step):
        mx, my = path_m[j]
        path_w.append(np.array(map_to_world(mx, my, map_size, map_res), dtype=float))
    if path_m:
        mx, my = path_m[-1]
        path_w.append(np.array(map_to_world(mx, my, map_size, map_res), dtype=float))
    return path_w
