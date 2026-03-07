# Drone Simulation Project

PyBullet-based drone simulation with maze navigation, occupancy mapping, and path planning (e.g. A*).

## Requirements

- Python 3.10+ (recommended)
- See [requirements.txt](requirements.txt) for full dependency list.

### Key dependencies

| Package | Purpose |
|--------|---------|
| **gym-pybullet-drones** | Drone sim environments (installed from GitHub) |
| **gymnasium** | RL environment API |
| **pybullet** | Physics engine |
| **stable_baselines3** | Reinforcement learning |
| **torch** | Deep learning |
| **numpy**, **scipy** | Numerics |
| **matplotlib**, **pandas** | Plotting and data |

## Installation

1. Clone the repo and enter the project directory:

   ```bash
   cd project
   ```

2. Create and activate a virtual environment (recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux/macOS
   # or: venv\Scripts\activate  # Windows
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   **Note:** `requirements.txt` includes `gym-pybullet-drones` as an editable install from GitHub. If you see a line like `packaging @ file:///...` (a local path), remove or replace it with `packaging` so installs work on other machines.

## Usage

Run the simulation (GUI, single drone, maze):

```bash
python sim.py
```

Adjust options at the top of `sim.py` (e.g. `GUI`, `NUM_DRONES`, `DURATION_SEC`, `MAP_SIZE`).

## Project structure

- **sim.py** – Main simulation: CtrlAviary, maze walls, occupancy map, path planning.
- **requirements.txt** – Python dependencies and versions.
- **gym-pybullet-drones/** – Submodule/library for drone environments and control (DSLPIDControl, etc.).

## License

See the `gym-pybullet-drones` subtree for its license. Project-specific code is provided as-is.
