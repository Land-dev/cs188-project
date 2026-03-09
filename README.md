# Drone Simulation Project

A PyBullet-based drone simulation where an autonomous Crazyflie 2.x explores a maze, detects fires via simulated lidar, and navigates to extinguish them — all while avoiding obstacles.

## Quick Start

```bash
git clone https://github.com/Land-dev/cs188-project.git
cd cs188-project
chmod +x install.sh && ./install.sh
conda activate drones
python sim.py
```

> **Prerequisites:** [conda](https://docs.conda.io/en/latest/miniconda.html) must be installed and available on your PATH.

## How It Works

1. **SLAM Exploration** — The drone casts 144 lidar rays to build an occupancy map (unknown → free/obstacle/fire) while flying frontier-based exploration goals.
2. **Fire Detection** — When lidar intersects a fire cylinder, the drone immediately replans a path to the fire.
3. **A\* Path Planning with Obstacle Inflation** — Paths are planned on the occupancy grid using A\* with a configurable safe margin (`safe_margin=0.25m`). Obstacles are inflated using `scipy.ndimage.binary_dilation` to keep the drone a safe distance from walls and pillars.
4. **Smooth PID Control** — A position-error clamp and tuned step size prevent aggressive maneuvers that could destabilize the drone.
5. **Multi-Fire** — After extinguishing a fire (hover for 2s), a new fire spawns randomly. The simulation ends when all fires are out (default: 3).

## Project Structure

| File | Description |
|---|---|
| `sim.py` | Main simulation loop — environment setup, SLAM, control, fire logic |
| `path_planning.py` | A\* planner with scipy-based obstacle inflation (soft cost margins) |
| `test_path_planning.py` | Unit tests for path planning (obstacle avoidance, fire reachability) |
| `run_batch.py` | Batch runner — runs N headless sims in parallel, reports success rate |
| `install.sh` | One-command setup: conda env, pybullet patch, dependencies |

## Running Tests

```bash
conda activate drones
python -m pytest test_path_planning.py -v
```

## Batch Testing

Run 10 headless simulations in parallel to measure reliability:

```bash
conda activate drones
python run_batch.py
```

**Latest result: 10/10 (100%) success rate, average 45s per run.**

## CLI Flags

| Flag | Description |
|---|---|
| `--headless` | Run without GUI (for batch testing / CI) |

Environment variable `SIM_SEED` sets the random seed for reproducible runs.

---

## Setup Details

### Install Script

The `install.sh` script handles:
- Cloning [gym-pybullet-drones](https://github.com/utiasDSL/gym-pybullet-drones)
- Creating the `drones` conda environment
- Patching and building pybullet from source (required on macOS Apple Silicon)
- Installing all remaining dependencies

## Known Issues

The two issues below affect **macOS Apple Silicon (M1/M2/M3)** users and are both handled automatically by `install.sh`.

---

### Issue 1: `pybullet` fails to build — clang exits with code 1

**How to identify it:** During `pip install -e .` inside `gym-pybullet-drones/`, the build fails with:

```
error: command '/usr/bin/clang' failed with exit code 1
ERROR: Failed building wheel for pybullet
```

**Root cause:** pybullet's bundled zlib defines `#define fdopen(fd, mode) NULL` whenever `TARGET_OS_MAC` is set, which clobbers the system `fdopen()` declaration on macOS 15.

**Manual fix:** Download the source, apply the `__APPLE__` guard, and install from the patched copy:

```bash
conda activate drones
mkdir /tmp/pybullet_patch && cd /tmp/pybullet_patch
pip download pybullet==3.2.6 --no-deps --no-binary :all: -d .
tar xzf pybullet-3.2.6.tar.gz && cd pybullet-3.2.6
python - examples/ThirdPartyLibs/zlib/zutil.h <<'EOF'
import sys; path = sys.argv[1]
with open(path) as f: src = f.read()
old = '#define fdopen(fd, mode) NULL /* No fdopen() */'
new = '#if !defined(__APPLE__)\n#define fdopen(fd, mode) NULL /* No fdopen() */\n#endif'
open(path, 'w').write(src.replace(old, new))
EOF
pip install .
```

Then install the rest from inside `gym-pybullet-drones/`:

```bash
cd <path-to-cs188-project>/gym-pybullet-drones
pip install -e . --no-deps
pip install numpy scipy transforms3d matplotlib pytest gymnasium "stable-baselines3>=2.0.0" control
```

---

### Issue 2: `No module named 'pkg_resources'`

**Root cause:** `pkg_resources` is part of `setuptools`. In `setuptools` 82+, it was removed. `gym-pybullet-drones` still imports it.

**Manual fix:**

```bash
conda activate drones
pip install "setuptools<81" --force-reinstall
```
