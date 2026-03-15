# SLAM-Based Firefighting Drone Simulation

A PyBullet-based drone simulation where an autonomous Crazyflie 2.x quadrotor performs Simultaneous Localization and Mapping (SLAM) in a hazardous indoor environment. The drone explores unknown layouts, builds an occupancy grid map via LiDAR sensing, and utilizes a custom **360° Panoramic Computer Vision** stack to locate and extinguish fire sources.

<p align="center">
  <img src="drone_cv_mission.gif" width="700" alt="Drone Mission Demo" />
</p>

## Project Overview

This project simulates the core autonomy stack for an aerial firefighting robot. Operating in a 6x6m indoor arena with complex pillar obstacles, the drone must cope with:
- **Sensor Uncertainty:** Position and LiDAR data are corrupted with Gaussian noise to simulate real-world hardware.
- **Autonomous Exploration:** A frontier-based strategy ensures the drone systematically maps the entire room.
- **Dynamic Path Planning:** An A* motion planner with 0.25m obstacle inflation keeps the drone safe from collisions.
- **Visual Intelligence:** A 4-camera panoramic stitcher and OpenCV pipeline provide constant 360° situational awareness for fire detection.

---


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

1. **Simulated Localization** — Optionally, the drone does not get perfect state: position is corrupted by Gaussian noise (e.g. GPS), and lidar ranges are perturbed. An EKF fuses the noisy position into an estimate; all planning, exploration, and map logic use this estimate. The drone must cope with both **where am I?** and **where are obstacles?** uncertainty (set `USE_LOCALIZATION = True` in `sim.py`).
2. **SLAM Exploration** — The drone casts 144 lidar rays to build an occupancy map (unknown → free/obstacle/fire) while flying frontier-based exploration goals.
3. **Fire Detection** — When lidar intersects a fire cylinder, the drone immediately replans a path to the fire.
4. **A\* Path Planning with Obstacle Inflation** — Paths are planned on the occupancy grid using A\* with a configurable safe margin (`safe_margin=0.25m`). Obstacles are inflated using `scipy.ndimage.binary_dilation` to keep the drone a safe distance from walls and pillars.
5. **Smooth PID Control** — A position-error clamp and tuned step size prevent aggressive maneuvers that could destabilize the drone.
6. **360° Panoramic Vision** — The drone captures 4 cardinal FPV views (90° FOV each) and stitches them into a panoramic strip. This allows for constant, 360-degree situational awareness.
7. **CV Fire Detection** — An OpenCV-based pipeline in `vision.py` scans the panoramic feed for specific color signatures (Orange/Red) corresponding to fire cylinders.
8. **Dynamic Yaw Control** — The drone dynamically adjusts its heading (yaw) to face its current waypoint, ensuring efficient movement and optimal camera coverage.
9. **Autonomous Navigation** — When a fire is confirmed by CV, the drone calculates the target coordinates via angular projection and plans an A* path to extinguish it.
10. **Multi-Fire Mission** — After extinguishing a fire (hover for 2s), a new one spawns randomly. The mission follows a 3-fire objective.

## Project Structure

| File | Description |
| :--- | :--- |
| `sim.py` | Main simulation loop — environment setup, SLAM, control, fire logic, optional localization |
| `localization.py` | EKF pose estimator and sensor noise (noisy position, optional lidar range noise) |
| `path_planning.py` | A* planner with scipy-based obstacle inflation (soft cost margins) |
| `run_batch.py` | Batch runner — runs N headless sims in parallel, reports success rate |
| `install.sh` | One-command setup: conda env, pybullet patch, dependencies |
| `vision.py` | Computer Vision module — HSV thresholding and contour detection logic |
| `test_cv.py` | Unit tests for the Computer Vision fire detection system |
| `test_path_planning.py` | Unit tests for navigation and obstacle avoidance logic |
| `record_mission.sh` | Convenience script to record mission highlights (requires `ffmpeg`) |

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
