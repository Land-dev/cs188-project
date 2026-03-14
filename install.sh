#!/usr/bin/env bash
# install.sh — sets up the full drone simulation environment.
# Run from the root of the cs188-project directory.
#
# Usage:
#   chmod +x install.sh
#   ./install.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRONES_DIR="$SCRIPT_DIR/gym-pybullet-drones"

# ── 1. Clone gym-pybullet-drones if not already present ───────────────────────
if [ ! -d "$DRONES_DIR/.git" ]; then
    echo ">>> Cloning gym-pybullet-drones..."
    git clone https://github.com/utiasDSL/gym-pybullet-drones.git "$DRONES_DIR"
else
    echo ">>> gym-pybullet-drones already cloned, skipping."
fi

# ── 2. Create conda environment ────────────────────────────────────────────────
if conda env list | grep -q "^drones "; then
    echo ">>> conda env 'drones' already exists, skipping creation."
else
    echo ">>> Creating conda env 'drones' with Python 3.10..."
    conda create -n drones python=3.10 -y
fi

# Resolve the pip/python inside the drones env without requiring conda activate
# (conda activate does not work reliably inside non-interactive scripts)
DRONES_PYTHON="$(conda run -n drones which python)"
DRONES_PIP="$(conda run -n drones which pip)"

echo ">>> Using Python: $DRONES_PYTHON"

# ── 3. Upgrade pip ────────────────────────────────────────────────────────────
echo ">>> Upgrading pip..."
"$DRONES_PIP" install --upgrade pip

# ── 4. Install pybullet from patched source (required on macOS Apple Silicon) ─
#
# pybullet has no pre-built ARM64 wheel on PyPI and must be compiled from source.
# Its bundled zlib (examples/ThirdPartyLibs/zlib/zutil.h) defines:
#
#     #define fdopen(fd, mode) NULL   /* No fdopen() */
#
# whenever TARGET_OS_MAC is set (always true on macOS). This macro clobbers the
# system fdopen() declaration in macOS 15's _stdio.h, causing a clang parse error.
# The patch guards that macro so it is skipped on Apple platforms, where fdopen
# has always been available.

PYBULLET_BUILD_DIR="/tmp/pybullet_patch_build"

echo ">>> Building pybullet from patched source..."
rm -rf "$PYBULLET_BUILD_DIR"
mkdir -p "$PYBULLET_BUILD_DIR"

"$DRONES_PIP" download pybullet==3.2.6 --no-deps --no-binary :all: -d "$PYBULLET_BUILD_DIR"
tar xzf "$PYBULLET_BUILD_DIR/pybullet-3.2.6.tar.gz" -C "$PYBULLET_BUILD_DIR"

ZUTIL="$PYBULLET_BUILD_DIR/pybullet-3.2.6/examples/ThirdPartyLibs/zlib/zutil.h"

# Apply the __APPLE__ guard to every occurrence of the problematic fdopen macro.
# We use Python for the replacement to avoid sed portability issues.
"$DRONES_PYTHON" - "$ZUTIL" <<'PYEOF'
import sys
path = sys.argv[1]
with open(path) as f:
    src = f.read()
old = '#define fdopen(fd, mode) NULL /* No fdopen() */'
new = '#if !defined(__APPLE__)\n#define fdopen(fd, mode) NULL /* No fdopen() */\n#endif'
patched = src.replace(old, new)
with open(path, 'w') as f:
    f.write(patched)
print(f"  Patched {src.count(old)} occurrence(s) in {path}")
PYEOF

"$DRONES_PIP" install "$PYBULLET_BUILD_DIR/pybullet-3.2.6"
rm -rf "$PYBULLET_BUILD_DIR"

# ── 5. Install gym-pybullet-drones (without reinstalling pybullet) ────────────
echo ">>> Installing gym-pybullet-drones..."
"$DRONES_PIP" install -e "$DRONES_DIR" --no-deps

# ── 6. Install remaining dependencies ─────────────────────────────────────────
echo ">>> Installing remaining dependencies..."
"$DRONES_PIP" install \
    numpy scipy transforms3d matplotlib pytest \
    gymnasium "stable-baselines3>=2.0.0" control opencv-python

# ── 7. Pin setuptools to a version that still exposes pkg_resources ───────────
#
# setuptools 82+ removed pkg_resources as a top-level importable module.
# gym-pybullet-drones's BaseAviary.py uses `import pkg_resources`, so we pin
# to <81 to keep it available. This is harmless — pkg_resources is deprecated
# but still functional at this version.
echo ">>> Pinning setuptools to restore pkg_resources..."
"$DRONES_PIP" install "setuptools<81" --force-reinstall

echo ""
echo "✓ Setup complete. To run the simulation:"
echo ""
echo "    conda activate drones"
echo "    python sim.py"
echo ""
