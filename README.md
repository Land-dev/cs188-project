# Drone Simulation Project

## Setup

### 1. Clone this repository

```bash
git clone https://github.com/Land-dev/cs188-project.git
cd cs188-project
```

### 2. Run the install script

The script handles everything: cloning [gym-pybullet-drones](https://github.com/utiasDSL/gym-pybullet-drones) into the correct subfolder, creating the `drones` conda environment, patching and building pybullet from source (required on macOS Apple Silicon — see [Known Issues](#known-issues) below), and installing all remaining dependencies.

```bash
chmod +x install.sh
./install.sh
```

> **Prerequisites:** [conda](https://docs.conda.io/en/latest/miniconda.html) must be installed and available on your PATH.

### 3. Run the simulation

```bash
conda activate drones
python sim.py
```

---

## Known Issues

The two issues below affect **macOS Apple Silicon (M1/M2/M3)** users and are both handled automatically by `install.sh`. They are documented here so you understand what the script does and can fix things manually if needed.

---

### Issue 1: `pybullet` fails to build — clang exits with code 1

**How to identify it:** During `pip install -e .` inside `gym-pybullet-drones/`, the build fails with:

```
error: command '/usr/bin/clang' failed with exit code 1
ERROR: Failed building wheel for pybullet
× Failed to build installable wheels for some pyproject.toml based projects
╰─> pybullet
```

The verbose error points to `_stdio.h:318: error: expected identifier or '('`.

**Root cause:** pybullet has no pre-built ARM64 wheel on PyPI and must compile from source. Its bundled zlib (`examples/ThirdPartyLibs/zlib/zutil.h`) defines:

```c
#define fdopen(fd, mode) NULL   /* No fdopen() */
```

whenever `TARGET_OS_MAC` is set — which is always true on macOS. This macro then clobbers the system `fdopen()` declaration in macOS 15's `_stdio.h`, causing clang to fail parsing the header.

**Manual fix:** Download the source, apply the `__APPLE__` guard, and install from the patched copy:

```bash
conda activate drones
mkdir /tmp/pybullet_patch && cd /tmp/pybullet_patch

pip download pybullet==3.2.6 --no-deps --no-binary :all: -d .
tar xzf pybullet-3.2.6.tar.gz
cd pybullet-3.2.6

# Guard the macro so it is skipped on macOS (which always has fdopen)
python - examples/ThirdPartyLibs/zlib/zutil.h <<'EOF'
import sys
path = sys.argv[1]
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

**How to identify it:** When running `python sim.py`:

```
ModuleNotFoundError: No module named 'pkg_resources'
```

Note: `pip install pkg_resources` will **not** work — it is not a standalone PyPI package.

**Root cause:** `pkg_resources` is part of `setuptools`. In `setuptools` 82+, it was removed as a top-level importable module. `gym-pybullet-drones`'s `BaseAviary.py` still uses `import pkg_resources`, so it breaks with the newer version.

**Manual fix:**

```bash
conda activate drones
pip install "setuptools<81" --force-reinstall
```

Verify it works:

```bash
python -c "import pkg_resources; print('OK')"
```

A deprecation warning may appear — it is harmless.
