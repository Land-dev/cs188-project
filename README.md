# Drone Simulation Project

A PyBullet-based drone simulation where an autonomous Crazyflie 2.x explores a maze, detects fires via **360° Panoramic Computer Vision**, and navigates to extinguish them — all while avoiding obstacles using lidar-based SLAM.

## Quick Start

```bash
git clone https://github.com/Land-dev/cs188-project.git
cd cs188-project
chmod +x install.sh && ./install.sh
conda activate drones
python sim.py
```

> **Prerequisites:** [conda](https://docs.conda.io/en/latest/miniconda.html) and [ffmpeg](https://ffmpeg.org/download.html) must be installed and available on your PATH.

## How It Works

1. **Lidar-based SLAM** — The drone casts 144 lidar rays to build an occupancy map for obstacle avoidance and exploration. Obstacles are inflated to ensure a safe margin.
2. **360° Panoramic Vision** — The drone captures 4 cardinal FPV views (90° FOV each) and stitches them into a panoramic strip. This allows for constant, 360-degree situational awareness.
3. **CV Fire Detection** — An OpenCV-based pipeline in `vision.py` scans the panoramic feed for specific color signatures (Orange/Red) corresponding to fire cylinders.
4. **Dynamic Yaw Control** — The drone dynamically adjusts its heading (yaw) to face its current waypoint, ensuring efficient movement and optimal camera coverage.
5. **Autonomous Navigation** — When a fire is confirmed by CV, the drone calculates the target coordinates via angular projection and plans an A* path to extinguish it.
6. **Multi-Fire Mission** — After extinguishing a fire (hover for 2s), a new one spawns randomly. The mission follows a 3-fire objective.

## Project Structure

| File | Description |
|---|---|
| `sim.py` | Main simulation loop — handles environment, 360° vision rig, and control |
| `vision.py` | Computer Vision module — HSV thresholding and contour detection logic |
| `path_planning.py` | A* planner with obstacle inflation and occupancy grid mapping |
| `test_cv.py` | Unit tests for the Computer Vision fire detection system |
| `test_path_planning.py` | Unit tests for navigation and obstacle avoidance logic |
| `record_mission.sh` | Convenience script to record mission highlights (requires `ffmpeg`) |
| `run_batch.py` | Parallel batch runner for measuring simulation reliability |
| `install.sh` | Automated setup script for dependencies and PyBullet patches |

## Running Tests

Verify the CV and navigation systems:

```bash
conda activate drones
python -m pytest test_cv.py -v
python -m pytest test_path_planning.py -v
```

## Recording Missions

Capture a highlight of the drone's 360° CV feed:

```bash
./record_mission.sh
```

This generates a `drone_cv_mission.mp4` file showing the panoramic FPV feed used by the vision algorithm.

## CLI Flags

| Flag | Description |
|---|---|
| `--headless` | Run without GUI (for batch testing / CI) |
| `--record`   | Enable video recording of the FPV feed |

Environment variable `SIM_SEED` sets the random seed for reproducible runs.

## Troubleshooting

The `install.sh` script automatically handles common issues on macOS Apple Silicon, such as `pybullet` build failures and `setuptools` version conflicts. See the script source for manual patch details if needed.
