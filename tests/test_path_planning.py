import numpy as np
from src.path_planning import plan_path, world_to_map, map_to_world

def test_plan_path_without_inflation():
    """Test that path planning without safe margin grazes the obstacle."""
    map_size = 6
    map_res = 0.1
    map_dim = int(map_size / map_res)
    occupancy_map = np.ones((map_dim, map_dim), dtype=np.uint8) * 128
    
    # Create an obstacle block from (0.0, -1.0) to (1.0, 1.0)
    for mx in range(map_dim):
        for my in range(map_dim):
            x, y = map_to_world(mx, my, map_size, map_res)
            if 0.0 <= x <= 1.0 and -1.0 <= y <= 1.0:
                occupancy_map[mx, my] = 255
    
    start_xy = (-1.0, 0.0)
    goal_xy = (2.0, 0.0)
    
    # With safe_margin=0.0, the path should graze the obstacle.
    path = plan_path(occupancy_map, map_size, map_res, start_xy, goal_xy, safe_margin=0.0)
    assert len(path) > 0, "A valid path should be found."
    
    # The path should get very close to the obstacle bounding box
    min_dist = np.inf
    for pt in path:
        x, y = pt
        dx = max(0.0, 0.0 - x, x - 1.0)
        dy = max(0.0, -1.0 - y, y - 1.0)
        dist = np.hypot(dx, dy)
        if dist < min_dist:
            min_dist = dist
    
    assert min_dist < 0.15, f"Path too far from obstacle without inflation! Dist: {min_dist}"

def test_plan_path_with_inflation():
    """Test that path planning with safe margin keeps a distance from the obstacle."""
    map_size = 6
    map_res = 0.1
    map_dim = int(map_size / map_res)
    occupancy_map = np.ones((map_dim, map_dim), dtype=np.uint8) * 128
    
    for mx in range(map_dim):
        for my in range(map_dim):
            x, y = map_to_world(mx, my, map_size, map_res)
            if 0.0 <= x <= 1.0 and -1.0 <= y <= 1.0:
                occupancy_map[mx, my] = 255
    
    start_xy = (-1.0, 0.0)
    goal_xy = (2.0, 0.0)
    
    safe_margin = 0.3
    path = plan_path(occupancy_map, map_size, map_res, start_xy, goal_xy, safe_margin=safe_margin)
    assert len(path) > 0, "A valid path should be found even with inflation."
    
    # Now the minimum distance to the obstacle bounding box should be >= safe_margin
    min_dist = np.inf
    for pt in path:
        x, y = pt
        dx = max(0.0, 0.0 - x, x - 1.0)
        dy = max(0.0, -1.0 - y, y - 1.0)
        dist = np.hypot(dx, dy)
        if dist < min_dist:
            min_dist = dist
            
    # Allow some tolerance for grid resolution/discretization artifacts
    assert min_dist >= safe_margin - map_res * 1.5, f"Path got too close with inflation! Dist: {min_dist}"

def test_path_to_fire_works():
    """Test that a path can be planned to a fire cell (val=200) even with inflation.
    Simulates realistic scenario where lidar marks multiple cells around fire as 200."""
    map_size = 6
    map_res = 0.1
    map_dim = int(map_size / map_res)
    occupancy_map = np.ones((map_dim, map_dim), dtype=np.uint8) * 128
    
    # Place fire cells in a small cluster around (1.0, 1.0), like lidar would
    fire_center = (1.0, 1.0)
    for dx in np.arange(-0.15, 0.2, 0.1):
        for dy in np.arange(-0.15, 0.2, 0.1):
            fmx, fmy = world_to_map(fire_center[0]+dx, fire_center[1]+dy, map_size, map_res)
            occupancy_map[fmx, fmy] = 200
    
    start_xy = (0.0, 0.0)
    goal_xy = fire_center
    
    path = plan_path(occupancy_map, map_size, map_res, start_xy, goal_xy, safe_margin=0.25)
    assert len(path) > 0, "Should find a path to the fire even with inflation."
    
    # Last waypoint should be AT the fire, not just nearby
    last = path[-1]
    dist_to_fire = np.hypot(last[0] - fire_center[0], last[1] - fire_center[1])
    assert dist_to_fire < 0.15, f"Last waypoint should be AT the fire, got dist={dist_to_fire:.3f}"
    
    # No waypoint should veer away from fire (monotonically approaching)
    # The path should not have the drone moving AWAY from fire near the end
    if len(path) >= 3:
        last_3_dists = [np.hypot(p[0]-fire_center[0], p[1]-fire_center[1]) for p in path[-3:]]
        assert last_3_dists[-1] <= last_3_dists[0] + 0.1, \
            f"Path veers away from fire at the end! Distances: {last_3_dists}"

if __name__ == "__main__":
    test_plan_path_without_inflation()
    print("PASS: test_plan_path_without_inflation")
    test_plan_path_with_inflation()
    print("PASS: test_plan_path_with_inflation")
    test_path_to_fire_works()
    print("PASS: test_path_to_fire_works")
    print("ALL TESTS PASSED")
